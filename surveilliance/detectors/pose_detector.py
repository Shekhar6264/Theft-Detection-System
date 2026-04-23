import os

import numpy as np


class PoseDetector:
    def __init__(self):
        self.min_keypoint_confidence = float(
            os.getenv("POSE_MIN_KEYPOINT_CONFIDENCE", "0.35")
        )
        self.min_vertical_ratio = float(os.getenv("POSE_MIN_VERTICAL_RATIO", "0.03"))
        self.min_extension_ratio = float(os.getenv("POSE_MIN_EXTENSION_RATIO", "0.08"))
        self.min_suspicion_score = float(os.getenv("POSE_MIN_SUSPICION_SCORE", "0.55"))

    def analyze(self, results, person_box=None, iou_threshold=0.3):
        best_match = {
            "detected": False,
            "score": 0.0,
            "side": None,
        }

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints_batch = result.keypoints.xy.cpu().numpy()
            confidence_batch = self._get_keypoint_confidences(result, keypoints_batch)

            for box, keypoints, confidences in zip(boxes, keypoints_batch, confidence_batch):
                if person_box is not None and self._iou(box, person_box) < iou_threshold:
                    continue

                analysis = self._analyze_person(keypoints, confidences, box)
                if analysis["score"] > best_match["score"]:
                    best_match = analysis

        return best_match

    def detect(self, results, person_box=None, iou_threshold=0.3):
        return self.analyze(results, person_box, iou_threshold)["detected"]

    def _get_keypoint_confidences(self, result, keypoints_batch):
        keypoint_confidences = getattr(result.keypoints, "conf", None)
        if keypoint_confidences is None:
            return np.ones(keypoints_batch.shape[:2], dtype=float)
        return keypoint_confidences.cpu().numpy()

    def _analyze_person(self, keypoints, confidences, person_box):
        person_height = max(1.0, float(person_box[3] - person_box[1]))
        side_scores = {
            "left": self._analyze_arm(
                shoulder=keypoints[5],
                elbow=keypoints[7],
                wrist=keypoints[9],
                shoulder_conf=confidences[5],
                elbow_conf=confidences[7],
                wrist_conf=confidences[9],
                person_height=person_height,
            ),
            "right": self._analyze_arm(
                shoulder=keypoints[6],
                elbow=keypoints[8],
                wrist=keypoints[10],
                shoulder_conf=confidences[6],
                elbow_conf=confidences[8],
                wrist_conf=confidences[10],
                person_height=person_height,
            ),
        }

        side, score = max(side_scores.items(), key=lambda item: item[1])
        return {
            "detected": score >= self.min_suspicion_score,
            "score": score,
            "side": side if score > 0 else None,
        }

    def _analyze_arm(
        self,
        shoulder,
        elbow,
        wrist,
        shoulder_conf,
        elbow_conf,
        wrist_conf,
        person_height,
    ):
        average_confidence = (
            float(shoulder_conf) + float(elbow_conf) + float(wrist_conf)
        ) / 3.0
        if (
            float(shoulder_conf) < self.min_keypoint_confidence * 0.8
            or float(wrist_conf) < self.min_keypoint_confidence * 0.7
            or average_confidence < self.min_keypoint_confidence
        ):
            return 0.0

        wrist_raise = max(0.0, float(shoulder[1] - wrist[1])) / person_height
        elbow_raise = max(0.0, float(shoulder[1] - elbow[1])) / person_height
        vertical_raise = max(wrist_raise, elbow_raise * 0.7)
        arm_extension = float(np.linalg.norm(np.array(wrist) - np.array(shoulder))) / person_height
        horizontal_extension = max(0.0, float(abs(wrist[0] - shoulder[0]))) / person_height

        if vertical_raise < self.min_vertical_ratio:
            return 0.0
        if wrist_raise <= 0 and elbow_raise <= 0:
            return 0.0
        if max(arm_extension, horizontal_extension) < self.min_extension_ratio:
            return 0.0

        confidence_score = min(1.0, average_confidence)
        raise_score = min(1.0, vertical_raise / max(self.min_vertical_ratio, 1e-6))
        elbow_score = min(1.0, elbow_raise / max(self.min_vertical_ratio * 0.6, 1e-6))
        extension_score = min(
            1.0,
            max(arm_extension, horizontal_extension) / max(self.min_extension_ratio, 1e-6),
        )

        return 0.35 * confidence_score + 0.35 * raise_score + 0.15 * elbow_score + 0.15 * extension_score

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
