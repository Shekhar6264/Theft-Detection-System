def calculate_threat(person_detected, weapon_detected):
    score = 0

    if person_detected:
        score += 2

    if weapon_detected:
        score += 5

    return score
