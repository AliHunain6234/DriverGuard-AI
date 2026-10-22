# DriverGuard AI

Real-time driver drowsiness monitoring: eye closure (EAR), yawning (MAR) and head-pose tracking with MediaPipe, a risk score, CSV logs and PDF session reports.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

`app.py` is a standalone OpenCV desktop version (press Q to quit).

## Deploy on Streamlit Cloud

- Main file: `dashboard.py`; Python version: 3.11.
- Add a TURN server in the app's Secrets so the browser camera can connect.
  Easiest: a free Metered account (metered.ca), then
  `[metered]` with `domain = "yourapp.metered.live"` and `api_key = "..."`.
  Or set a fixed server:

```toml
[turn]
urls = ["turn:YOUR_HOST:443?transport=tcp"]
username = "..."
credential = "..."
```

Risk scores are heuristic estimates, not a validated measure of fitness to drive.
