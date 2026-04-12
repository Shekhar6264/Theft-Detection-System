import os
import cv2
import torch
from ultralytics import YOLO

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        if not self.video.isOpened():
            print("❌ Camera not accessible")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 🔥 UPDATE THIS IF train2/train3 EXISTS
        weapon_model_path = os.path.join(
            BASE_DIR,
            "weapon_dataset",
            "runs",
            "detect",
            "train",   # 🔁 change if needed
            "weights",
            "best.pt"
        )

        print("Weapon model path:", weapon_model_path)

        # 🔥 DEVICE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("🔥 Using device:", self.device)

        # 🔥 LOAD MODELS
        self.person_model = YOLO("yolov8n.pt")
        self.weapon_model = YOLO(weapon_model_path)

        self.person_model.to(self.device)
        self.weapon_model.to(self.device)

        print("✅ Models loaded")

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        success, frame = self.video.read()

        if not success:
            return b''

        frame = cv2.resize(frame, (416, 320))
        display = frame.copy()

        # =========================
        # 🔵 PERSON DETECTION
        # =========================
        person_results = self.person_model(frame, conf=0.4, device=self.device)

        if person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                cls = int(box.cls[0])

                # ✅ ONLY PERSON
                if cls != 0:
                    continue

                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                height = y2 - y1
                if height < 80:
                    continue

                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display, f"Person {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)

        # =========================
        # 🔴 WEAPON DETECTION
        # =========================
        weapon_results = self.weapon_model(
            frame,
            conf=0.5,
            device=self.device
        )

        if weapon_results[0].boxes is not None:
            for box in weapon_results[0].boxes:
                conf = float(box.conf[0])

                if conf < 0.6:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 🔥 ONLY "WEAPON" LABEL
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display, f"Weapon {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', display)
        return jpeg.tobytes() if ret else b''