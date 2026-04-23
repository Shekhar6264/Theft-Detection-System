import os


class WeaponDetector:
    def __init__(self):
        configured_classes = os.getenv("WEAPON_CLASSES", "knife,guns")
        self.weapon_classes = {
            label.strip().lower()
            for label in configured_classes.split(",")
            if label.strip()
        }
        self.min_confidence = float(os.getenv("WEAPON_CONFIDENCE_THRESHOLD", "0.55"))
        self.instant_lock_confidence = float(
            os.getenv("WEAPON_INSTANT_LOCK_CONFIDENCE", "0.75")
        )
        self.min_box_area_ratio = float(os.getenv("WEAPON_MIN_BOX_AREA_RATIO", "0.003"))
        self.max_box_area_ratio = float(os.getenv("WEAPON_MAX_BOX_AREA_RATIO", "0.35"))

    def analyze(self, results, model, image_shape):
        image_height, image_width = image_shape[:2]
        image_area = max(1, image_height * image_width)
        best_match = {
            "detected": False,
            "confidence": 0.0,
            "label": None,
            "area_ratio": 0.0,
            "instant_lock": False,
        }

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                if confidence < self.min_confidence:
                    continue

                cls = int(box.cls[0])
                label = str(model.names[cls]).lower()
                if label not in self.weapon_classes:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_ratio = box_area / image_area
                if area_ratio < self.min_box_area_ratio:
                    continue
                if self.max_box_area_ratio > 0 and area_ratio > self.max_box_area_ratio:
                    continue

                if confidence > best_match["confidence"]:
                    best_match = {
                        "detected": True,
                        "confidence": confidence,
                        "label": label,
                        "area_ratio": area_ratio,
                        "instant_lock": confidence >= self.instant_lock_confidence,
                    }

        return best_match

    def detect(self, results, model, image_shape):
        return self.analyze(results, model, image_shape)["detected"]
