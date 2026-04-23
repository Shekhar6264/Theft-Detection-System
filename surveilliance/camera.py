import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# 🔥 DETECTORS
from detectors.person_detector import PersonDetector
from detectors.weapon_detector import WeaponDetector
from detectors.pose_detector import PoseDetector
from detectors.mask_detector import MaskDetector

# 🔥 LOGIC
from logic.activity_logic import get_activity
from logic.threat_score import calculate_threat, get_threat_level
from utils.email_alert import EmailAlertService
from utils.monitoring_rules import MonitoringRules


class VideoCamera:
    def __init__(self):
        self.camera_source = self._parse_camera_source(os.getenv("CAMERA_SOURCE", "0"))
        self.video = cv2.VideoCapture(self.camera_source)

        if not self.video.isOpened():
            print(f"❌ Camera not accessible: {self.camera_source}")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = BASE_DIR

        weapon_model_path = os.path.join(
            BASE_DIR, "weapon_dataset", "runs", "detect", "train", "weights", "best.pt"
        )

        mask_model_path = os.path.join(
            BASE_DIR, "mask_dataset", "runs", "detect", "train", "weights", "best.pt"
        )

        # 🔥 Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("🔥 Using device:", self.device)

        # 🔥 Models
        self.person_model = YOLO("yolov8n.pt")
        self.weapon_model = YOLO(weapon_model_path)
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.mask_model = YOLO(mask_model_path)

        self.person_model.to(self.device)
        self.weapon_model.to(self.device)
        self.pose_model.to(self.device)
        self.mask_model.to(self.device)

        # 🔥 Detectors
        self.person_detector = PersonDetector()
        self.weapon_detector = WeaponDetector()
        self.pose_detector = PoseDetector()
        self.mask_detector = MaskDetector()

        # 🔥 Frame control
        self.frame_count = 0
        self.fps = self._detect_fps()
        self.detection_interval = max(1, int(os.getenv("DETECTION_INTERVAL_FRAMES", "2")))
        self.weapon_interval = max(1, int(os.getenv("WEAPON_INTERVAL_FRAMES", "2")))
        self.pose_interval = max(self.detection_interval, int(os.getenv("POSE_INTERVAL_FRAMES", "2")))
        self.mask_interval = max(self.detection_interval, int(os.getenv("MASK_INTERVAL_FRAMES", "3")))
        self.alert_threshold = int(os.getenv("THREAT_ALERT_THRESHOLD", "80"))
        self.alert_cooldown = int(os.getenv("THREAT_ALERT_COOLDOWN_SECONDS", "60"))
        self.last_alert_time = 0.0
        self.clip_seconds = int(os.getenv("ALERT_CLIP_SECONDS", "5"))
        self.frame_buffer = deque(maxlen=max(1, self.fps * self.clip_seconds))
        self.alert_dir = Path(BASE_DIR) / "surveilliance" / "alerts"
        self.alert_service = EmailAlertService(self.alert_dir)
        self.monitoring_rules = MonitoringRules()
        self.person_state = {}
        self.confirmation_frames = max(1, int(os.getenv("CONFIRMATION_FRAMES", "2")))
        self.weapon_confirmation_frames = max(
            self.confirmation_frames,
            int(os.getenv("WEAPON_CONFIRMATION_FRAMES", "3")),
        )
        self.weapon_signal_threshold = float(
            os.getenv("WEAPON_SIGNAL_THRESHOLD", "1.0")
        )
        self.weapon_signal_decay = float(os.getenv("WEAPON_SIGNAL_DECAY", "0.35"))
        self.pose_signal_threshold = float(os.getenv("POSE_SIGNAL_THRESHOLD", "1.2"))
        self.pose_signal_decay = float(os.getenv("POSE_SIGNAL_DECAY", "0.45"))
        self.mask_signal_threshold = float(os.getenv("MASK_SIGNAL_THRESHOLD", "0.6"))
        self.mask_signal_decay = float(os.getenv("MASK_SIGNAL_DECAY", "0.35"))
        self.min_pose_person_height = int(os.getenv("POSE_MIN_PERSON_HEIGHT", "90"))
        self.min_pose_person_width = int(os.getenv("POSE_MIN_PERSON_WIDTH", "48"))
        self.latest_status = {
            "system_mode": "MONITOR",
            "monitoring_active": self.monitoring_rules.is_monitoring_active(),
            "active_schedule": self.monitoring_rules.describe_schedule(),
            "zone_enabled": self.monitoring_rules.zone_enabled,
            "people_count": 0,
            "top_activity": "No Person Detected",
            "top_score": 0,
            "top_reasons": [],
            "pipeline_mode": "BALANCED",
        }

        print("✅ System Ready")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def _parse_camera_source(self, source):
        try:
            return int(source)
        except (TypeError, ValueError):
            return source

    def _detect_fps(self):
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 10

    def _crop_with_padding(self, frame, box, pad_x=0.08, pad_y=0.08):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        x1 = max(0, int(x1 - box_w * pad_x))
        y1 = max(0, int(y1 - box_h * pad_y))
        x2 = min(width, int(x2 + box_w * pad_x))
        y2 = min(height, int(y2 + box_h * pad_y))
        return frame[y1:y2, x1:x2]

    def _crop_face_region(self, frame, box):
        x1, y1, x2, y2 = box
        box_h = y2 - y1
        face_y2 = y1 + int(box_h * 0.68)
        return self._crop_with_padding(frame, (x1, y1, x2, face_y2), pad_x=0.16, pad_y=0.18)

    def _crop_pose_region(self, frame, box):
        return self._crop_with_padding(frame, box, pad_x=0.22, pad_y=0.18)

    def _update_streak(self, previous_value, detected):
        if detected:
            return previous_value + 1
        return max(0, previous_value - 1)

    def _update_signal(self, previous_value, evidence_score, decay, boost=1.0, maximum=3.0):
        if evidence_score > 0:
            return min(maximum, max(0.0, previous_value * 0.55) + (evidence_score * boost))
        return max(0.0, previous_value - decay)

    def _format_mask_status(self, mask_status):
        labels = {
            "mask": "Mask On",
            "incorrect_mask": "Mask Incorrect",
            "no_mask": "No Mask",
            "unknown": "Mask Unknown",
        }
        return labels.get(mask_status, "Mask Unknown")

    def _mask_signal_boost(self, mask_status, confidence):
        if mask_status == "no_mask":
            return confidence
        if mask_status == "incorrect_mask":
            return confidence * 1.1
        if mask_status == "mask":
            return confidence
        return 0.0

    def _draw_person_panel(self, display, box, person_id, activity, level, score, color, status_lines):
        x1, y1, x2, y2 = box
        panel_x1 = x1
        panel_x2 = min(display.shape[1] - 8, x1 + 220)
        line_height = 18
        panel_height = 26 + (len(status_lines) + 2) * line_height
        panel_y2 = max(8 + panel_height, y1 - 8)
        panel_y1 = max(8, panel_y2 - panel_height)

        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (18, 24, 36), -1)
        cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)
        cv2.rectangle(display, (panel_x1, panel_y1), (panel_x2, panel_y2), color, 2)

        header = f"ID {person_id}  {level} ({score})"
        cv2.putText(display, header,
                    (panel_x1 + 10, panel_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        cv2.putText(display, activity,
                    (panel_x1 + 10, panel_y1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        for index, line in enumerate(status_lines):
            text_y = panel_y1 + 58 + (index * line_height)
            cv2.putText(display, line,
                        (panel_x1 + 10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 225, 235), 1)

    def _draw_global_hud(self, display, monitoring_active, people_count, top_level):
        zone_color = (0, 200, 255) if monitoring_active else (120, 120, 120)
        mode_text = "ACTIVE" if monitoring_active else "STANDBY"
        schedule_text = self.monitoring_rules.describe_schedule()

        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (display.shape[1] - 10, 52), (12, 18, 28), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.putText(display, f"Theft Detection Dashboard  |  Mode: {mode_text}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"People: {people_count}  |  Schedule: {schedule_text}  |  State: {top_level}",
                    (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)

    def _draw_protected_zone(self, display):
        if not self.monitoring_rules.zone_enabled:
            return

        zx1, zy1, zx2, zy2 = self.monitoring_rules.get_zone_pixels(display.shape)
        overlay = display.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (0, 165, 255), -1)
        cv2.addWeighted(overlay, 0.08, display, 0.92, 0, display)
        cv2.rectangle(display, (zx1, zy1), (zx2, zy2), (0, 165, 255), 2)
        cv2.putText(display, "Protected Zone",
                    (zx1 + 8, max(zy1 + 18, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    def _draw_zoom_preview(self, display, person_crop, box, color):
        if person_crop.size == 0:
            return

        preview_width = 88
        preview_height = 112
        preview = cv2.resize(person_crop, (preview_width, preview_height))
        x1, y1, x2, _ = box
        preview_x1 = min(max(8, x2 + 10), max(8, display.shape[1] - preview_width - 8))
        preview_y1 = max(8, y1)
        preview_x2 = preview_x1 + preview_width
        preview_y2 = min(display.shape[0] - 8, preview_y1 + preview_height)
        actual_height = preview_y2 - preview_y1
        actual_width = preview_x2 - preview_x1
        display[preview_y1:preview_y2, preview_x1:preview_x2] = preview[:actual_height, :actual_width]
        cv2.rectangle(display, (preview_x1, preview_y1), (preview_x2, preview_y2), color, 2)
        cv2.putText(display, "Focus",
                    (preview_x1 + 6, preview_y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    def _save_alert_clip(self):
        if not self.frame_buffer:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = self.alert_dir / f"threat_{timestamp}.mp4"
        first_frame = self.frame_buffer[0]
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )

        if not writer.isOpened():
            return None

        for buffered_frame in self.frame_buffer:
            writer.write(buffered_frame)

        writer.release()
        return clip_path

    def _save_snapshot(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.alert_dir / f"threat_{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return snapshot_path

    def _save_person_focus(self, frame, person_box):
        focus_crop = self._crop_with_padding(frame, person_box, pad_x=0.2, pad_y=0.15)
        if focus_crop.size == 0:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        focus_path = self.alert_dir / f"focus_{timestamp}.jpg"
        cv2.imwrite(str(focus_path), focus_crop)
        return focus_path

    def _send_alert_if_needed(self, score, level, activity, reasons, annotated_frame, person_box, raw_frame):
        now = time.time()

        if score < self.alert_threshold:
            return

        if now - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = now
        snapshot_path = self._save_snapshot(annotated_frame)
        focus_path = self._save_person_focus(raw_frame, person_box)
        clip_path = self._save_alert_clip()
        self.alert_service.send_alert(
            threat_score=score,
            threat_level=level,
            activity=activity,
            reasons=reasons,
            camera_source=str(self.camera_source),
            snapshot_path=snapshot_path,
            focus_path=focus_path,
            clip_path=clip_path,
        )

    def get_status(self):
        return {
            **self.latest_status,
            "email_configured": self.alert_service.is_configured(),
        }

    def get_email_config(self):
        return self.alert_service.get_public_config()

    def update_email_config(self, **kwargs):
        self.alert_service.update_config(**kwargs)
        return self.alert_service.get_public_config()

    def get_frame(self):
        success, frame = self.video.read()

        if not success:
            return b''

        frame = cv2.resize(frame, (416, 320))
        display = frame.copy()
        self.frame_buffer.append(frame.copy())
        monitoring_active = self.monitoring_rules.is_monitoring_active()

        # 🔥 Frame skipping (performance)
        self.frame_count += 1
        run_weapon_detection = (self.frame_count % self.weapon_interval == 0)
        run_pose_detection = (self.frame_count % self.pose_interval == 0)
        run_mask_detection = (self.frame_count % self.mask_interval == 0)

        # =========================
        # 🔵 PERSON TRACKING
        # =========================
        person_results = self.person_model.track(
            frame,
            persist=True,
            conf=0.4,
            device=self.device
        )

        tracked_persons = []

        if person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                if int(box.cls[0]) != 0:
                    continue

                if box.id is None:
                    continue

                person_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                tracked_persons.append((person_id, x1, y1, x2, y2))

        top_score = 0
        top_level = "MONITOR"
        top_activity = "No Person Detected"
        top_reasons = []

        self._draw_protected_zone(display)
        self._draw_global_hud(display, monitoring_active, len(tracked_persons), top_level)

        # =========================
        # 🔁 PER PERSON ANALYSIS
        # =========================
        for person_id, x1, y1, x2, y2 in tracked_persons:
            person_box = (x1, y1, x2, y2)
            current_in_protected_zone = self.monitoring_rules.is_in_protected_zone(
                person_box, frame.shape
            )
            person_crop = self._crop_with_padding(frame, person_box, pad_x=0.08, pad_y=0.08)
            pose_crop = self._crop_pose_region(frame, person_box)
            face_crop = self._crop_face_region(frame, person_box)
            if person_crop.size == 0:
                continue

            state = self.person_state.get(
                person_id,
                {
                    "weapon_detected": False,
                    "mask_status": "unknown",
                    "mask_confidence": 0.0,
                    "mask_signal": 0.0,
                    "hand_raised": False,
                    "weapon_streak": 0,
                    "weapon_confidence": 0.0,
                    "weapon_signal": 0.0,
                    "pose_streak": 0,
                    "pose_signal": 0.0,
                    "in_protected_zone": False,
                },
            )
            weapon_detected = state["weapon_detected"]
            mask_status = state["mask_status"]
            mask_confidence = state.get("mask_confidence", 0.0)
            mask_signal = state.get("mask_signal", 0.0)
            hand_raised = state["hand_raised"]
            weapon_streak = state.get("weapon_streak", 0)
            weapon_confidence = state.get("weapon_confidence", 0.0)
            weapon_signal = state.get("weapon_signal", 0.0)
            pose_streak = state.get("pose_streak", 0)
            pose_signal = state.get("pose_signal", 0.0)
            in_protected_zone = current_in_protected_zone

            should_run_weapon = run_weapon_detection or weapon_signal > 0.2 or in_protected_zone
            should_run_mask = run_mask_detection or mask_signal > 0.2 or in_protected_zone
            person_height = y2 - y1
            person_width = x2 - x1
            should_run_pose = (
                run_pose_detection
                and person_height >= self.min_pose_person_height
                and person_width >= self.min_pose_person_width
                and pose_crop.size != 0
            )

            if should_run_weapon or should_run_mask or should_run_pose:
                raw_hand_raised = False
                pose_analysis = {"detected": False, "score": 0.0, "side": None}
                if should_run_pose:
                    pose_results = self.pose_model(
                        pose_crop,
                        device=self.device,
                        verbose=False,
                    )
                    pose_analysis = self.pose_detector.analyze(
                        pose_results
                    )
                    raw_hand_raised = pose_analysis["detected"]

                raw_weapon_detected = False
                instant_weapon_lock = False

                if should_run_weapon and person_crop.shape[0] >= 80 and person_crop.shape[1] >= 50:
                    weapon_results = self.weapon_model(
                        person_crop,
                        conf=self.weapon_detector.min_confidence,
                        device=self.device,
                        verbose=False,
                    )
                    weapon_analysis = self.weapon_detector.analyze(
                        weapon_results, self.weapon_model, person_crop.shape
                    )
                    raw_weapon_detected = weapon_analysis["detected"]
                    weapon_confidence = weapon_analysis["confidence"]
                    instant_weapon_lock = weapon_analysis["instant_lock"]
                elif should_run_weapon:
                    raw_weapon_detected = False
                    weapon_confidence = 0.0

                if should_run_mask and face_crop.size != 0 and face_crop.shape[0] >= 40 and face_crop.shape[1] >= 40:
                    mask_results = self.mask_model(
                        face_crop,
                        conf=self.mask_detector.min_confidence,
                        device=self.device,
                        verbose=False,
                    )
                    mask_analysis = self.mask_detector.analyze(
                        mask_results, self.mask_model
                    )
                    mask_signal = self._update_signal(
                        mask_signal,
                        self._mask_signal_boost(
                            mask_analysis["status"],
                            mask_analysis["confidence"],
                        ),
                        self.mask_signal_decay,
                        boost=1.0,
                        maximum=2.5,
                    )
                    if mask_signal >= self.mask_signal_threshold:
                        mask_status = mask_analysis["status"]
                        mask_confidence = mask_analysis["confidence"]
                    else:
                        mask_status = (
                            mask_analysis["status"]
                            if mask_analysis["confidence"] >= self.mask_detector.min_confidence
                            else "unknown"
                        )
                        mask_confidence = (
                            mask_analysis["confidence"] if mask_status != "unknown" else 0.0
                        )
                elif should_run_mask:
                    mask_status = "unknown"
                    mask_confidence = 0.0
                    mask_signal = max(0.0, mask_signal - self.mask_signal_decay)

                weapon_streak = self._update_streak(weapon_streak, raw_weapon_detected)
                pose_streak = self._update_streak(pose_streak, raw_hand_raised)
                weapon_signal = self._update_signal(
                    weapon_signal,
                    weapon_confidence if raw_weapon_detected else 0.0,
                    self.weapon_signal_decay,
                    boost=1.2,
                )
                pose_signal = self._update_signal(
                    pose_signal,
                    pose_analysis["score"] if raw_hand_raised else 0.0,
                    self.pose_signal_decay,
                    boost=0.9,
                )
                weapon_detected = instant_weapon_lock or (
                    weapon_signal >= self.weapon_signal_threshold
                    or weapon_streak >= self.weapon_confirmation_frames
                )
                hand_raised = (
                    pose_signal >= self.pose_signal_threshold
                    and pose_streak >= self.confirmation_frames
                )

                self.person_state[person_id] = {
                    "weapon_detected": weapon_detected,
                    "mask_status": mask_status,
                    "mask_confidence": mask_confidence,
                    "mask_signal": mask_signal,
                    "hand_raised": hand_raised,
                    "weapon_streak": weapon_streak,
                    "weapon_confidence": weapon_confidence,
                    "weapon_signal": weapon_signal,
                    "pose_streak": pose_streak,
                    "pose_signal": pose_signal,
                    "in_protected_zone": in_protected_zone,
                }

            # =========================
            # 🧠 ACTIVITY
            # =========================
            activity = get_activity(
                True,
                in_protected_zone,
                monitoring_active,
                weapon_detected,
                hand_raised,
                mask_status,
                mask_confidence,
            )

            # =========================
            # 🚨 THREAT SCORE
            # =========================
            score, reasons = calculate_threat(
                True,
                in_protected_zone,
                monitoring_active,
                mask_status,
                mask_confidence,
                weapon_detected,
                hand_raised,
                activity,
            )

            level, color = get_threat_level(score)

            # =========================
            # 🎯 DRAW CLEAN UI
            # =========================
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            status_lines = [
                "Room status: person detected",
                f"Zone: {'inside protected area' if in_protected_zone else 'outside protected area'}",
                f"Weapon: {'detected' if weapon_detected else 'verifying' if weapon_signal > 0.2 else 'clear'}"
                + (f" ({weapon_confidence:.2f})" if weapon_confidence > 0 else ""),
                f"Pose: {'suspicious movement' if hand_raised else 'verifying' if pose_signal > 0.2 else 'normal'}",
                self._format_mask_status(mask_status)
                + (f" ({mask_confidence:.2f})" if mask_confidence > 0 else ""),
            ]
            self._draw_person_panel(
                display,
                person_box,
                person_id,
                activity,
                level,
                score,
                color,
                status_lines,
            )
            self._draw_zoom_preview(display, person_crop, person_box, color)

            reason_y = min(y2 + 22, display.shape[0] - 8)
            if reasons:
                cv2.putText(display, reasons[0],
                            (x1, reason_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 244, 200), 1)

            if score > top_score:
                top_score = score
                top_level = level
                top_activity = activity
                top_reasons = reasons

            self._send_alert_if_needed(score, level, activity, reasons, display, person_box, frame)

        self._draw_global_hud(display, monitoring_active, len(tracked_persons), top_level)

        active_ids = {person_id for person_id, _, _, _, _ in tracked_persons}
        self.person_state = {
            person_id: state
            for person_id, state in self.person_state.items()
            if person_id in active_ids
        }

        self.latest_status = {
            "system_mode": top_level,
            "monitoring_active": monitoring_active,
            "active_schedule": self.monitoring_rules.describe_schedule(),
            "zone_enabled": self.monitoring_rules.zone_enabled,
            "people_count": len(tracked_persons),
            "top_activity": top_activity,
            "top_score": top_score,
            "top_reasons": top_reasons,
            "pipeline_mode": "BALANCED",
        }

        # =========================
        # 🖼 ENCODE
        # =========================
        ret, jpeg = cv2.imencode('.jpg', display)
        return jpeg.tobytes() if ret else b''
