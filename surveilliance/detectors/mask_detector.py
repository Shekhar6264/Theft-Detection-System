import os


class MaskDetector:
    def __init__(self):
        self.mask_labels = ["incorrectly_worn_mask", "with_mask", "without_mask"]
        self.min_confidence = float(os.getenv("MASK_CONFIDENCE_THRESHOLD", "0.35"))

    def analyze(self, results, model):
        best_match = {
            "status": "unknown",
            "confidence": 0.0,
        }
        priority = {
            "no_mask": 3,
            "incorrect_mask": 2,
            "mask": 1,
            "unknown": 0,
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
                if label == "without_mask":
                    status = "no_mask"
                elif label == "incorrectly_worn_mask":
                    status = "incorrect_mask"
                elif label == "with_mask":
                    status = "mask"
                else:
                    continue

                if (
                    priority[status] > priority[best_match["status"]]
                    or (
                        priority[status] == priority[best_match["status"]]
                        and confidence > best_match["confidence"]
                    )
                ):
                    best_match = {
                        "status": status,
                        "confidence": confidence,
                    }

        return best_match

    def detect(self, results, model):
        return self.analyze(results, model)["status"]
