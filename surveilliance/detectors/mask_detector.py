class MaskDetector:
    def __init__(self):
        self.mask_labels = ["incorrectly_worn_mask", "with_mask", "without_mask"]

    def detect(self, results, model):
        mask_status = "unknown"

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "without_mask":
                    return "no_mask"

                elif label == "incorrectly_worn_mask":
                    return "incorrect_mask"

                elif label == "with_mask":
                    mask_status = "mask"

        return mask_status