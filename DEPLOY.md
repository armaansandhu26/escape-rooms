# Deploy for teammates

This app is a **Python web server** (not static-only), so **Render** is the simplest free host. Vercel is built for static sites and serverless functions; this project would need a rewrite to run there.

## Render (recommended)

1. Push this repo to GitHub (see below).
2. Go to [render.com](https://render.com) and sign in with GitHub.
3. **New → Blueprint** and select this repository (Render reads `render.yaml`).
4. Deploy. Share the URL Render gives you (e.g. `https://escape-rooms-xxxx.onrender.com`).

Free tier may sleep after inactivity; the first visit can take ~30s to wake up.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/play_web.py
```

Open `http://127.0.0.1:8765`.
