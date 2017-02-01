"""Merge sort module."""

"""

MERGE SORT (MS)
===============

CodeFellows 401d5
Submission Date:

Authors:    Colin Lamont <https://github.com/chamberi>
            Ben Shields <https://github.com/iamrobinhood12345>

URL:    https://github.com/chamberi/data-structures/tree/merge-sort

"""


def merge_sort(sort_list):
    """Merge sort method."""
    if len(sort_list) < 2:
        return sort_list
    sort_list1 = sort_list[:int(len(sort_list) / 2)]
    sort_list2 = sort_list[int(len(sort_list) / 2):]

    sort_list1 = merge_sort(sort_list1)
    sort_list2 = merge_sort(sort_list2)

    def _merge(a, b):
        """Merge compares the two lists and returns a sorted list from lowest to highest value.""" 
        sorted_list = []
        while len(a) and len(b):
            if a[0] < b[0]:
                low = a[0]
                a = a[1:]
            else:
                low = b[0]
                b = b[1:]
            sorted_list.append(low)
        if len(a):
            sorted_list.extend(a)
            return sorted_list
        else:
            sorted_list.extend(b)
            return sorted_list

    return _merge(sort_list1, sort_list2)


def _random_list(n):
    """Return a list of random numbers from 0 to 300 of size n."""
    import random
    b = random
    return b.sample(range(0, 300), n)

a = _random_list(150)
r = a[:]
b = sorted(a)
w = b[::-1]


if __name__ == "__main__":
    import timeit

    random_merge_sort_timed = timeit.repeat(stmt="merge_sort(r)", setup="from merge_sort import merge_sort, r", number=1000, repeat=3)
    random_average_merge_sort_timed = float(sum(random_merge_sort_timed) / len(random_merge_sort_timed))

    print("number of runs: " + str(3))
    print("random merge_sort_timed: " + str(random_merge_sort_timed))
    print("average: ", str(random_average_merge_sort_timed))

    best_merge_sort_timed = timeit.repeat(stmt="merge_sort(b)", setup="from merge_sort import merge_sort, b", number=1000, repeat=3)
    best_average_merge_sort_timed = float(sum(best_merge_sort_timed) / len(best_merge_sort_timed))

    print("number of runs: " + str(3))
    print("best case merge_sort_timed: " + str(best_merge_sort_timed))
    print("average: ", str(best_average_merge_sort_timed))

    reverse_merge_sort_timed = timeit.repeat(stmt="merge_sort(w)", setup="from merge_sort import merge_sort, w", number=1000, repeat=3)
    reverse_average_merge_sort_timed = float(sum(reverse_merge_sort_timed) / len(reverse_merge_sort_timed))

    print("number of runs: " + str(3))
    print("reverse case merge_sort_timed: " + str(reverse_merge_sort_timed))
    print("average: ", str(reverse_average_merge_sort_timed))
