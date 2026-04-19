class WeaponDetector:
    def __init__(self):
        self.weapon_classes = ["knife"]

    def detect(self, results, model, min_confidence=0.35):
        weapon_detected = False

        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                if confidence < min_confidence:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]

                if label in self.weapon_classes:
                    weapon_detected = True

        return weapon_detected
