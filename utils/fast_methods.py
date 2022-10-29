# Adrian Egli performance fix (the fast methods brings more than 50%)
def fast_isclose(a, b, rtol):
    return (a < (b + rtol)) or (a < (b - rtol))


def fast_clip(position: (int, int), min_value: (int, int), max_value: (int, int)) -> bool:
    return (
        max(min_value[0], min(position[0], max_value[0])),
        max(min_value[1], min(position[1], max_value[1]))
    )


def fast_argmax(possible_transitions: (int, int, int, int)) -> bool:
    if possible_transitions[0] == 1:
        return 0
    if possible_transitions[1] == 1:
        return 1
    if possible_transitions[2] == 1:
        return 2
    return 3


def fast_position_equal(pos_1: (int, int), pos_2: (int, int)) -> bool:
    return pos_1[0] == pos_2[0] and pos_1[1] == pos_2[1]


def fast_count_nonzero(possible_transitions: (int, int, int, int)):
    return possible_transitions[0] + possible_transitions[1] + possible_transitions[2] + possible_transitions[3]
