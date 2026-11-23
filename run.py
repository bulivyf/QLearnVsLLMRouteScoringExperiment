"""Launch Route Recall Lab and open it in the default browser."""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="local port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="start only the server")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    host = "127.0.0.1"
    url = f"http://{host}:{args.port}/"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", host, "--port", str(args.port)],
        cwd=Path(__file__).resolve().parent,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if server.poll() is not None:
                return server.returncode
            try:
                with urlopen(url + "api/scenarios", timeout=1) as response:
                    if response.status == 200:
                        # Give Uvicorn a chance to report a bind failure if the
                        # probe reached an existing copy of the app.
                        time.sleep(0.2)
                        if server.poll() is not None:
                            return server.returncode
                        print(f"Route Recall Lab: {url}", flush=True)
                        if not args.no_browser and not webbrowser.open(url):
                            print("Open the URL above in your browser.", flush=True)
                        return server.wait()
            except (URLError, TimeoutError):
                time.sleep(0.15)
        print(f"App did not become ready at {url}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
