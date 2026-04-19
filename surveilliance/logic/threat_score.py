def calculate_threat(
    person_detected,
    in_protected_zone,
    monitoring_active,
    mask_status,
    weapon_detected,
    hand_raised,
    activity,
):
    """
    Returns a theft-risk score with a human-readable reason breakdown.
    """

    if not person_detected:
        return 0, []

    score = 10
    reasons = ["Person detected in monitored room"]

    if in_protected_zone:
        score += 18
        reasons.append("Person entered protected zone")
    else:
        score += 4

    if monitoring_active:
        score += 15
        reasons.append("Detection happened during active monitoring hours")
    else:
        reasons.append("Outside protected hours, continuing to monitor")

    if weapon_detected:
        score += 24
        reasons.append("Possible weapon detected")

    if hand_raised:
        score += 10
        reasons.append("Suspicious body pose detected")

    if mask_status == "mask":
        score += 8
        reasons.append("Face covered with mask")
    elif mask_status == "incorrect_mask":
        score += 12
        reasons.append("Face partially concealed")

    if activity == "Possible Theft In Progress":
        score += 12
        reasons.append("Combined activity pattern suggests theft")
    elif activity == "Armed Intruder Suspected":
        score += 8
    elif activity == "Unauthorized Presence In Protected Zone":
        score += 10
    elif activity == "Masked Person In Room":
        score += 6
    elif activity == "Face Concealment Detected":
        score += 8
    elif activity == "Suspicious Movement":
        score += 4

    if weapon_detected and hand_raised:
        score += 10
        reasons.append("Weapon and pose signals confirm each other")

    if weapon_detected and mask_status in {"mask", "incorrect_mask"}:
        score += 8
        reasons.append("Weapon and face covering raise theft confidence")

    if in_protected_zone and monitoring_active:
        score += 8
        reasons.append("Intrusion matched protected zone and schedule")

    score = max(0, min(100, score))
    return score, reasons[:4]


def get_threat_level(score):
    if score >= 75:
        return "THEFT ALERT", (0, 0, 255)
    if score >= 45:
        return "SUSPICIOUS", (0, 165, 255)
    return "MONITOR", (0, 255, 0)
