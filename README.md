# Centauri Networks Unified Pilot Website

One Flask application combining:
- `/` — Centauri Networks public/company website
- `/network` — all-cities DOOH monitoring dashboard
- `/ai` — live AI audience analytics, dwell time and QR interaction dashboard
- `/video_feed`, `/count`, `/qr_scan` — AI pilot endpoints

## Run
1. Put `yolov8n.pt` in this folder.
2. `python -m pip install -r requirements.txt`
3. `python detect.py`
4. Open `http://127.0.0.1:5000`

Do not use VS Code Live Server for this version.

## Important
The public site and dashboards are unified at one domain when deployed. The AI camera feed itself must run on the machine connected to the camera (or an edge device) and publish metrics/feed to the web backend for remote production use.
