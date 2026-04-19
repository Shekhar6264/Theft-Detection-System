import os
from datetime import datetime


class MonitoringRules:
    def __init__(self):
        self.zone_enabled = os.getenv("PROTECTED_ZONE_ENABLED", "1") == "1"
        self.zone = self._parse_zone(
            os.getenv("PROTECTED_ZONE", "0.20,0.12,0.82,0.95")
        )
        self.start_hour = int(os.getenv("ACTIVE_START_HOUR", "20"))
        self.end_hour = int(os.getenv("ACTIVE_END_HOUR", "6"))

    def _parse_zone(self, raw_zone):
        try:
            x1, y1, x2, y2 = [float(value.strip()) for value in raw_zone.split(",")]
            return (
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)),
                max(0.0, min(1.0, y2)),
            )
        except (TypeError, ValueError):
            return (0.20, 0.12, 0.82, 0.95)

    def is_monitoring_active(self, current_time=None):
        now = current_time or datetime.now()
        hour = now.hour

        if self.start_hour == self.end_hour:
            return True

        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour

        return hour >= self.start_hour or hour < self.end_hour

    def get_zone_pixels(self, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = self.zone
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )

    def is_in_protected_zone(self, person_box, frame_shape, min_overlap=0.25):
        if not self.zone_enabled:
            return True

        zx1, zy1, zx2, zy2 = self.get_zone_pixels(frame_shape)
        px1, py1, px2, py2 = person_box

        inter_x1 = max(zx1, px1)
        inter_y1 = max(zy1, py1)
        inter_x2 = min(zx2, px2)
        inter_y2 = min(zy2, py2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        person_area = max(0, px2 - px1) * max(0, py2 - py1)

        if person_area <= 0:
            return False

        return (inter_area / person_area) >= min_overlap

    def describe_schedule(self):
        return f"{self.start_hour:02d}:00-{self.end_hour:02d}:00"
