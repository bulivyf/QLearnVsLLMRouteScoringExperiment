# Route Recall Lab

A standalone FastAPI comparison of structured and LLM-assisted historical
recall on fictional civilian shipping routes. The eleven scripted incidents
contrast cases with the same structured category and different road impacts.

## Run

From the extracted route_recall_lab directory:

    python -m pip install -r requirements.txt
    python run.py

The launcher opens http://127.0.0.1:8000 once the app is ready. Use
`python run.py --port 8080` for another port or add `--no-browser` to skip
opening a tab. Press Ctrl+C to stop the server.

Set OPENAI_API_KEY in the launch terminal to enable
LLM-assisted scoring. ROUTE_RECALL_MODEL optionally overrides gpt-5-mini.
Each comparison sends its synthetic incident and candidate cases to OpenAI.
Without credentials, the other two policies run and the LLM score is blank.

- Structured recall matches lane and category, prioritizes the same affected road,
  then ranks by historical reward. Ties favor the higher historical dispatch index.
- LLM-assisted recall receives exactly the same candidate cases, reads their
  notes alongside the current report, and cites one case. The backend validates
  the cited case and route.
- Q-learning uses lane, category, and the affected-road field. It trains on
  synthetic historical reward observations for all three actions per state, then
  freezes the values before evaluation. Each dispatch ends an episode, so the
  update has no next-state term. It estimates immediate expected dispatch reward.
  The evaluation reuses disruption types but uses separate travel-time profiles.
- Always Direct picks Direct independently of history.

The evaluator withholds the simulated route outcomes from the decision policies.
Each earns +100 for arrival within seven hours, +80 for a late arrival, or -50
for a blocked route, minus route cost. Full comparison totals use only incidents
where all four produced a result. Running a scenario again replaces its
score, so it cannot inflate the totals. The case library is persisted in
cases.sqlite3 and seeded on first comparison. Browser scores are session-page
state and clear on reload. The Q-learning arm receives simulated rewards for alternative routes during
training. Recall arms receive historical case records. This difference matters
when interpreting the comparison. Both recall arms see the same candidate set. The synthetic incidents
are teaching examples, not evidence of expected performance in live logistics.


## Distinguishing incidents

The route diagrams show an intermediate node on each path (Depot, Junction,
or Ridge). The simulation scores the entire route and does not optimize
individual segments. The added incidents include mild and severe congestion
on the same reported road, and roadworks where a secondary delay can make
the usual bypass a poor choice. The three-field state intentionally omits
severity and secondary effects. The in-app Methods guide explains fit,
limitations, and compute requirements. Focus or hover a question mark for
a short explanation of a specific policy.
