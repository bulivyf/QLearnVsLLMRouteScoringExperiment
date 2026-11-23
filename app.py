import json
import os
import sqlite3
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Route Recall Lab")
ROOT = Path(__file__).resolve().parent
DB = Path(os.getenv("ROUTE_RECALL_DB", ROOT / "cases.sqlite3"))
ROUTES = {"Direct": (4, 4), "Bypass": (6, 10), "Outer": (7, 13)}
LANES = ("A → D", "B → E", "C → F")
WAYPOINTS = {"Direct": "Depot", "Bypass": "Junction", "Outer": "Ridge"}

# Synthetic historical precedents and their observed road profiles.
PATTERNS = (
    ("Congestion", "Routine school traffic. The direct road normally clears before dispatch.", "Direct", 96, (4, 6, 7)),
    ("Congestion", "A disabled delivery truck is blocking the direct road. The bypass and outer road are clear.", "Bypass", 90, (None, 6, 7)),
    ("Roadworks", "A signal upgrade has narrowed the direct road. The bypass is open.", "Bypass", 90, (10, 6, 7)),
    ("Roadworks", "Resurfacing has closed the bypass. The direct and outer roads are open.", "Direct", 96, (4, None, 7)),
    ("Weather", "Snow has delayed the direct road and bypass. The outer road was plowed.", "Outer", 87, (11, 10, 7)),
    ("Weather", "Heavy rain flooded the outer road. Direct and bypass remain open.", "Direct", 96, (4, 6, None)),
    ("Bridge closure", "Scheduled bridge inspection closed Direct. Bypass remained open.", "Bypass", 90, (None, 6, 7)),
    ("Congestion", "Brief queue at Depot. Direct finished in five hours.", "Direct", 96, (5, 6, 7)),
    ("Congestion", "Short delivery queue at Depot. Direct remained the fastest route.", "Direct", 96, (6, 7, 7)),
    ("Roadworks", "Full resurfacing at Depot closed Direct and delayed Bypass. Outer was clear.", "Outer", 87, (None, 9, 7)),
    ("Roadworks", "A temporary Depot gate closure blocked Direct. Bypass was clear.", "Bypass", 90, (None, 6, 7)),
)
AFFECTED_ROADS = ("None", "Direct", "Direct", "Bypass", "Direct, Bypass", "Outer", "Direct",
                  "Direct", "Direct", "Direct", "Direct")
# Separate evaluation profiles. The same incident types recur, but arrival
# times can differ from the historical experiences used for training.
EVALUATION_HOURS = ((None, 6, 7), (5, None, 7), (12, 11, 7), (5, 6, 7),
                    (10, 7, 7), (5, 6, None), (None, 7, 7),
                    (5, 7, 7), (None, 10, 7), (None, 6, 7), (10, 6, 7))
SCENARIOS = [
    {"id": i, "lane": LANES[i % 3], "category": PATTERNS[p][0], "report": report,
     "affected_road": AFFECTED_ROADS[p], "hours": EVALUATION_HOURS[i]}
    for i, (p, report) in enumerate((
        (1, "Dispatch notes: a broken-down van is obstructing Direct. Bypass has normal traffic."),
        (3, "Paving crews closed Bypass for the day. Direct is running normally."),
        (4, "Snow slowed Direct and Bypass. The county has already cleared Outer."),
        (0, "School pickup congestion ended before the driver was due to leave."),
        (2, "Temporary signals on Direct are adding six hours. Bypass remains clear."),
        (5, "Standing water has shut Outer. The short direct road remains open."),
        (6, "The bridge authority closed Direct for an inspection. Bypass remains open."),
        (7, "Brief queue at the Direct depot. Current dispatch estimate is five hours on Direct."),
        (9, "Full resurfacing closed Direct at the depot. Bypass has a lengthy queue. Outer via the ridge is clear."),
        (10, "A temporary gate failure closed Direct at the depot. Bypass is running normally."),
        (2, "Temporary traffic signals at the depot make Direct a ten-hour trip. Bypass is clear."),
    ))
]


def retrieve(scenario):
    with closing(sqlite3.connect(DB)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY, lane TEXT, category TEXT, note TEXT,
                route TEXT, reward INTEGER, dispatch_index INTEGER
            )""")
            columns = {column["name"] for column in conn.execute("PRAGMA table_info(cases)")}
            if "affected_road" not in columns:
                conn.execute("ALTER TABLE cases ADD COLUMN affected_road TEXT")
            conn.executemany(
                """INSERT INTO cases (id, lane, category, note, route, reward, dispatch_index, affected_road)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET affected_road = excluded.affected_road
                   WHERE cases.affected_road IS NULL""",
                [(f"H{lane_index + 1}-{index + 1}", lane, category, note, route,
                  reward, index, AFFECTED_ROADS[index])
                 for lane_index, lane in enumerate(LANES)
                 for index, (category, note, route, reward, _) in enumerate(PATTERNS)],
            )
        rows = conn.execute(
            """SELECT id, lane, category, affected_road, note, route, reward FROM cases
               WHERE lane = ? AND category = ?
               ORDER BY (affected_road = ?) DESC, reward DESC, dispatch_index DESC, id LIMIT 4""",
            (scenario["lane"], scenario["category"], scenario["affected_road"]),
        )
        return [dict(row) for row in rows]


class Recall(BaseModel):
    route: Literal["Direct", "Bypass", "Outer"]
    case_id: str
    reason: str


def assisted(scenario, cases):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Choose a route in a fictional civilian delivery simulator. Direct, Bypass, "
         "and Outer are route names. Compare the current report with the candidate cases. "
         "Choose the route of a supplied case that best applies to this report, considering "
         "which roads are affected, travel time and cost. Return its ID and explain the match."),
        ("human", "{evidence}"),
    ])
    model = ChatOpenAI(model=os.getenv("ROUTE_RECALL_MODEL", "gpt-5-mini"),
                       timeout=35, max_retries=0)
    chain = prompt | model.with_structured_output(Recall, method="json_schema")
    return chain.invoke({"evidence": json.dumps({
        "lane": scenario["lane"], "category": scenario["category"],
        "report": scenario["report"], "affected_road": scenario["affected_road"], "cases": cases, "route_hours_and_cost": ROUTES, "waypoints": WAYPOINTS,
    })})


def score(scenario, route):
    index = tuple(ROUTES).index(route)
    _, cost = ROUTES[route]
    hours = scenario["hours"][index]
    outcome, reward = ("Blocked", -50) if hours is None else (
        ("On time", 100) if hours <= 7 else ("Late", 80)
    )
    return {"route": route, "hours": hours, "outcome": outcome, "reward": reward - cost}


@lru_cache(maxsize=1)
def train_q():
    """Train on synthetic historical dispatches. Each dispatch is terminal."""
    rng = Random(42)
    profiles = {}
    for lane in LANES:
        for i, (category, _note, _route, _reward, hours) in enumerate(PATTERNS):
            state = (lane, category, AFFECTED_ROADS[i])
            profiles.setdefault(state, []).append(hours)
    values = {}
    for state, history in profiles.items():
        # A balanced replay lends each historical profile equal weight.
        observations = [(action, score({"hours": hours}, action)["reward"])
                        for hours in history for action in ROUTES] * 40
        rng.shuffle(observations)
        estimates = dict.fromkeys(ROUTES, 0.0)
        visits = dict.fromkeys(ROUTES, 0)
        for action, reward in observations:
            visits[action] += 1
            estimates[action] += (reward - estimates[action]) / visits[action]
        values[state] = estimates
    return values


def q_learning(scenario):
    state = (scenario["lane"], scenario["category"], scenario["affected_road"])
    estimates = train_q()[state]
    route = max(estimates, key=estimates.get)
    return {**score(scenario, route),
            "q_values": {action: round(value, 2) for action, value in estimates.items()},
            "reason": "Highest learned Q-value for this lane, category, and affected road."}


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": [{k: v for k, v in s.items() if k != "hours"} for s in SCENARIOS],
            "routes": ROUTES, "waypoints": WAYPOINTS, "llm_enabled": bool(os.getenv("OPENAI_API_KEY")),
            "model": os.getenv("ROUTE_RECALL_MODEL", "gpt-5-mini")}


@app.post("/api/compare/{scenario_id}")
def compare(scenario_id: int):
    if not 0 <= scenario_id < len(SCENARIOS):
        raise HTTPException(404, "Scenario not found.")
    scenario = SCENARIOS[scenario_id]
    cases = retrieve(scenario)
    first = cases[0]
    result = {
        "cases": cases,
        "structured": {**score(scenario, first["route"]), "case_id": first["id"],
                       "reason": ("Matched lane, category, and affected road, then ranked by historical reward."
                                  if first["affected_road"] == scenario["affected_road"] else
                                  "No affected-road match. Ranked lane and category matches by historical reward.")},
        "q_learning": q_learning(scenario), "direct": score(scenario, "Direct"), "llm": None,
        "llm_error": "Set OPENAI_API_KEY and restart to run the LLM policy.",
    }
    if not os.getenv("OPENAI_API_KEY"):
        return result
    try:
        choice = assisted(scenario, cases)
    except Exception as error:
        status = getattr(error, "status_code", None)
        detail = f"HTTP {status}" if isinstance(status, int) else type(error).__name__
        result["llm_error"] = f"LLM request failed ({detail}). Check model access, quota, or connectivity and retry."
        return result
    if not isinstance(choice, Recall) or not any(
        case["id"] == choice.case_id and case["route"] == choice.route for case in cases
    ):
        result["llm_error"] = "LLM response did not cite a valid case for its route. Retry the comparison."
        return result
    result["llm"] = {**score(scenario, choice.route), **choice.model_dump()}
    result["llm_error"] = None
    return result
