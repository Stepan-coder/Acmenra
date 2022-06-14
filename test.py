from typing import Any, List, Dict


def __dif_lists_index(list_a: list, list_b: list) -> List[int]:
    diff = []
    for i in range(min(len(list_a), len(list_b))):
        if list_a[i] != list_b[i]:
            diff.append(i)
    return diff

print(equels(list_a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             list_b=[1, 2, 3, 5, 6, 7, 9, 0, 1, 2]))


