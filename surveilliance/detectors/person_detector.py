class PersonDetector:
    def detect(self, results, model):
        person_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "person":
                    person_detected = True

        return person_detected
