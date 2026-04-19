import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


class EmailAlertService:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.smtp_server = os.getenv("ALERT_SMTP_SERVER", "")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.sender_email = os.getenv("ALERT_SENDER_EMAIL", "")
        self.sender_password = os.getenv("ALERT_SENDER_PASSWORD", "")
        self.receiver_email = os.getenv("ALERT_RECEIVER_EMAIL", "")

    def is_configured(self):
        required = [
            self.smtp_server,
            self.sender_email,
            self.sender_password,
            self.receiver_email,
        ]
        return all(required)

    def send_alert(
        self,
        threat_score,
        threat_level,
        activity,
        reasons,
        camera_source,
        snapshot_path=None,
        focus_path=None,
        clip_path=None,
    ):
        if not self.is_configured():
            print("⚠️ Email alert skipped: SMTP configuration is incomplete.")
            return False

        message = EmailMessage()
        message["Subject"] = f"Theft Detection Alert - {threat_level} threat"
        message["From"] = self.sender_email
        message["To"] = self.receiver_email

        detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason_lines = reasons or ["Person detected in protected room"]
        message.set_content(
            "\n".join(
                [
                    "A theft detection alert was triggered by the CCTV monitoring system.",
                    f"Detected at: {detected_at}",
                    f"Threat level: {threat_level}",
                    f"Threat score: {threat_score}",
                    f"Activity: {activity}",
                    f"Camera source: {camera_source}",
                    "Reasons:",
                    *[f"- {reason}" for reason in reason_lines],
                    "",
                    "Attached evidence:",
                    "- Full CCTV snapshot from the alert frame",
                    "- Zoomed person image for identification",
                    "- Short CCTV clip captured from the recent video buffer",
                ]
            )
        )

        self._attach_file(message, snapshot_path, "image/jpeg")
        self._attach_file(message, focus_path, "image/jpeg")
        self._attach_file(message, clip_path, "video/mp4")

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            print(f"📧 Alert email sent to {self.receiver_email}")
            return True
        except Exception as exc:
            print(f"❌ Failed to send alert email: {exc}")
            return False

    def _attach_file(self, message, file_path, mime_type):
        if not file_path:
            return

        path = Path(file_path)
        if not path.exists():
            return

        maintype, subtype = mime_type.split("/", 1)
        with path.open("rb") as file_handle:
            message.add_attachment(
                file_handle.read(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
