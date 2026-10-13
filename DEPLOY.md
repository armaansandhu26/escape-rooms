# Deploy on Vercel (free, no card for hobby)

1. Push the latest code to GitHub: [escape-rooms](https://github.com/armaansandhu26/escape-rooms).
2. Go to [vercel.com](https://vercel.com) → **Add New → Project**.
3. Import the **escape-rooms** repo.
4. Leave defaults; Vercel reads `vercel.json` (`buildCommand` copies the UI into `public/`, installs Python deps).
5. Click **Deploy**. Share the `*.vercel.app` URL with teammates.

The demo uses **serverless API routes** under `api/` and keeps game state in a **session token** returned on each API call (required on Vercel).

## Local dev (unchanged)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/play_web.py
```

Open `http://127.0.0.1:8765`.
