class WeaponDetector:
    def __init__(self):
        self.weapon_classes = ["knife"]

    def detect(self, results, model):
        weapon_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in self.weapon_classes:
                    weapon_detected = True

        return weapon_detected
