class PoseDetector:
    def detect(self, results, person_box=None, iou_threshold=0.3):
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            keypoints_batch = r.keypoints.xy.cpu().numpy()

            for box, keypoints in zip(boxes, keypoints_batch):
                if person_box is not None and self._iou(box, person_box) < iou_threshold:
                    continue

                if self._has_raised_hand(keypoints):
                    return True

        return False

    def _has_raised_hand(self, keypoints):
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        return left_wrist[1] < left_shoulder[1] or right_wrist[1] < right_shoulder[1]

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union
