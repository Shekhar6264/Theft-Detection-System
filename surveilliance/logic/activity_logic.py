def get_activity(
    person_detected,
    in_protected_zone,
    monitoring_active,
    weapon_detected,
    hand_raised,
    mask_status="unknown",
):
    """
    Returns a theft-focused summary of the observed activity.
    """

    if not person_detected:
        return "No Person Detected"

    if weapon_detected and hand_raised:
        return "Possible Theft In Progress"

    if weapon_detected:
        return "Armed Intruder Suspected"

    if mask_status == "incorrect_mask":
        return "Face Concealment Detected"

    if mask_status == "mask":
        return "Masked Person In Room"

    if hand_raised:
        return "Suspicious Movement"

    if in_protected_zone and monitoring_active:
        return "Unauthorized Presence In Protected Zone"

    if in_protected_zone:
        return "Person Inside Protected Zone"

    return "Person Detected In Room"
