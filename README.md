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

## API Endpoints

- `GET /health`: simple health check for backend/frontend integration
- `GET /status`: current monitoring summary for the dashboard
- `GET /config/email`: current email-alert configuration without exposing the password
- `POST /config/email`: update SMTP settings at runtime from a future login/settings flow

## Important Environment Variables

Configure these in your terminal before running:

```bash
CAMERA_SOURCE=0
DETECTION_INTERVAL_FRAMES=2
WEAPON_INTERVAL_FRAMES=2
POSE_INTERVAL_FRAMES=4
MASK_INTERVAL_FRAMES=3
CONFIRMATION_FRAMES=2
WEAPON_CONFIRMATION_FRAMES=3
WEAPON_CONFIDENCE_THRESHOLD=0.55
WEAPON_INSTANT_LOCK_CONFIDENCE=0.75
WEAPON_MIN_BOX_AREA_RATIO=0.003
WEAPON_MAX_BOX_AREA_RATIO=0.35
WEAPON_SIGNAL_THRESHOLD=1.0
WEAPON_SIGNAL_DECAY=0.35
POSE_MIN_KEYPOINT_CONFIDENCE=0.45
POSE_MIN_VERTICAL_RATIO=0.04
POSE_MIN_EXTENSION_RATIO=0.10
POSE_MIN_SUSPICION_SCORE=0.65
POSE_SIGNAL_THRESHOLD=1.2
POSE_SIGNAL_DECAY=0.45
POSE_MIN_PERSON_HEIGHT=110
POSE_MIN_PERSON_WIDTH=60
MASK_CONFIDENCE_THRESHOLD=0.45
MASK_SIGNAL_THRESHOLD=0.9
MASK_SIGNAL_DECAY=0.35
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
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=0
```

Example email settings payload for future frontend/backend integration:

```json
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "sender_email": "user@example.com",
  "sender_password": "app_password",
  "receiver_email": "owner@example.com"
}
```

## Demo Talking Points

- Why theft-focused instead of generic threat-focused logic
- Protected zone and schedule reduce false alerts
- Separate weapon confirmation and confidence thresholds reduce false positives
- Explainable reasons make the decision transparent
- Email alert provides evidence for verification

## Main Files

- `surveilliance/camera.py`: live pipeline, overlay, alert trigger
- `surveilliance/logic/activity_logic.py`: theft-focused activity labels
- `surveilliance/logic/threat_score.py`: explainable theft-risk scoring
- `surveilliance/utils/email_alert.py`: email alert service
- `surveilliance/utils/monitoring_rules.py`: schedule and protected-zone rules
