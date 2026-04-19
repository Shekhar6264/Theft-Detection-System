# Smart Theft Detection System

Mini project for a CSE undergraduate surveillance use case. The system monitors a room using a CCTV feed, detects people, checks whether they enter a protected zone, analyzes weapon visibility, mask usage, and suspicious pose, then computes an explainable theft-risk score. When the score crosses the alert threshold, it captures evidence and emails the owner.

## Features

- Real-time person detection and tracking
- Protected-zone based intrusion monitoring
- Schedule-aware monitoring hours
- Weapon, mask, and suspicious pose analysis
- Explainable theft score with reason breakdown
- Zoomed focus preview for detected person
- Email alert with full frame, focus crop, and short CCTV clip
- Flask dashboard for live monitoring

## Project Flow

1. Camera/CCTV stream is read.
2. Person is detected and tracked.
3. The system checks whether the person is inside the protected zone.
4. Weapon, mask, and pose detectors run on the person crop.
5. Theft-risk score and reasons are generated.
6. If the score reaches alert threshold, evidence is saved and emailed.

## Run

```bash
pip install -r requirements.txt
cd surveilliance
python main.py
```

Open `http://127.0.0.1:5000`.

## Important Environment Variables

Configure these in your terminal before running:

```bash
CAMERA_SOURCE=0
DETECTION_INTERVAL_FRAMES=2
CONFIRMATION_FRAMES=2
THREAT_ALERT_THRESHOLD=80
THREAT_ALERT_COOLDOWN_SECONDS=60
ALERT_CLIP_SECONDS=5

PROTECTED_ZONE_ENABLED=1
PROTECTED_ZONE=0.20,0.12,0.82,0.95
ACTIVE_START_HOUR=20
ACTIVE_END_HOUR=6

ALERT_SMTP_SERVER=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SENDER_EMAIL=your_email@example.com
ALERT_SENDER_PASSWORD=your_app_password
ALERT_RECEIVER_EMAIL=owner_email@example.com
```

## Demo Talking Points

- Why theft-focused instead of generic threat-focused logic
- Protected zone and schedule reduce false alerts
- Confirmation frames reduce one-frame false positives
- Explainable reasons make the decision transparent
- Email alert provides evidence for verification

## Main Files

- `surveilliance/camera.py`: live pipeline, overlay, alert trigger
- `surveilliance/logic/activity_logic.py`: theft-focused activity labels
- `surveilliance/logic/threat_score.py`: explainable theft-risk scoring
- `surveilliance/utils/email_alert.py`: email alert service
- `surveilliance/utils/monitoring_rules.py`: schedule and protected-zone rules
