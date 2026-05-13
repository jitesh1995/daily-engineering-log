"""
Binary Search Variants
Common patterns beyond the basic binary search.
"""

def binary_search(arr, target):
    """Standard binary search. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr, target):
    """First position where arr[i] >= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(arr, target):
    """First position where arr[i] > target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search_rotated(arr, target):
    """Search in a rotated sorted array."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[lo] <= arr[mid]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def find_peak(arr):
    """Find a peak element (greater than neighbors)."""
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < arr[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search_answer(lo, hi, is_feasible):
    """
    Template for binary search on the answer space.
    Find minimum value where is_feasible(x) is True.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if is_feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Search 7: index={binary_search(arr, 7)}")
    print(f"Lower bound 6: index={lower_bound(arr, 6)}")
    print(f"Upper bound 7: index={upper_bound(arr, 7)}")

    rotated = [7, 9, 11, 13, 15, 1, 3, 5]
    print(f"Search rotated 3: index={search_rotated(rotated, 3)}")

    peaks = [1, 3, 7, 5, 2, 4, 1]
    print(f"Peak at index: {find_peak(peaks)}")
