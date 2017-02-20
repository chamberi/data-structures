"""Test module for k nearest neighbor."""

import pytest

TRAIN_DTC_DATA = [[1.4, 0.2, "setosa"], [1.4, 0.2, "setosa"], [1.3, 0.2, "setosa"], [1.5, 0.2, "setosa"], [1.4, 0.2, "setosa"], [1.7, 0.4, "setosa"], [1.4, 0.3, "setosa"], [1.5, 0.2, "setosa"], [1.4, 0.2, "setosa"], [1.5, 0.1, "setosa"], [1.5, 0.2, "setosa"], [1.6, 0.2, "setosa"], [1.4, 0.1, "setosa"], [1.1, 0.1, "setosa"], [1.2, 0.2, "setosa"], [1.5, 0.4, "setosa"], [1.3, 0.4, "setosa"], [1.4, 0.3, "setosa"], [1.7, 0.3, "setosa"], [1.5, 0.3, "setosa"], [1.7, 0.2, "setosa"], [1.5, 0.4, "setosa"], [1, 0.2, "setosa"], [1.7, 0.5, "setosa"], [1.9, 0.2, "setosa"], [1.6, 0.2, "setosa"], [1.6, 0.4, "setosa"], [1.5, 0.2, "setosa"], [1.4, 0.2, "setosa"], [1.6, 0.2, "setosa"], [1.6, 0.2, "setosa"], [1.5, 0.4, "setosa"], [1.5, 0.1, "setosa"], [1.4, 0.2, "setosa"], [1.5, 0.1, "setosa"], [1.2, 0.2, "setosa"], [1.3, 0.2, "setosa"], [1.5, 0.1, "setosa"], [1.3, 0.2, "setosa"], [1.5, 0.2, "setosa"], [3.5, 1, "versicolor"], [4.2, 1.5, "versicolor"], [4, 1, "versicolor"], [4.7, 1.4, "versicolor"], [3.6, 1.3, "versicolor"], [4.4, 1.4, "versicolor"], [4.5, 1.5, "versicolor"], [4.1, 1, "versicolor"], [4.5, 1.5, "versicolor"], [3.9, 1.1, "versicolor"], [4.8, 1.8, "versicolor"], [4, 1.3, "versicolor"], [4.9, 1.5, "versicolor"], [4.7, 1.2, "versicolor"], [4.3, 1.3, "versicolor"], [4.4, 1.4, "versicolor"], [4.8, 1.4, "versicolor"], [5, 1.7, "versicolor"], [4.5, 1.5, "versicolor"], [3.5, 1, "versicolor"], [3.8, 1.1, "versicolor"], [3.7, 1, "versicolor"], [3.9, 1.2, "versicolor"], [5.1, 1.6, "versicolor"], [4.5, 1.5, "versicolor"], [4.5, 1.6, "versicolor"], [4.7, 1.5, "versicolor"], [4.4, 1.3, "versicolor"], [4.1, 1.3, "versicolor"], [4, 1.3, "versicolor"], [4.4, 1.2, "versicolor"], [4.6, 1.4, "versicolor"], [4, 1.2, "versicolor"], [3.3, 1, "versicolor"], [4.2, 1.3, "versicolor"], [4.2, 1.2, "versicolor"], [4.2, 1.3, "versicolor"], [4.3, 1.3, "versicolor"], [3, 1.1, "versicolor"], [4.1, 1.3, "versicolor"]]

TEST_DTC_DATA = [[[[1.3, 0.3]], ["setosa"]], [[[1.3, 0.3]], ["setosa"]], [[[1.3, 0.2]], ["setosa"]], [[[1.6, 0.6]], ["setosa"]], [[[1.9, 0.4]], ["setosa"]], [[[1.4, 0.3]], ["setosa"]], [[[1.6, 0.2]], ["setosa"]], [[[1.4, 0.2]], ["setosa"]], [[[1.5, 0.2]], ["setosa"]], [[[1.4, 0.2]], ["setosa"]], [[[4.7, 1.4]], ["versicolor"]], [[[4.5, 1.5]], ["versicolor"]], [[[4.9, 1.5]], ["versicolor"]], [[[4, 1.3]], ["versicolor"]], [[[4.6, 1.5]], ["versicolor"]], [[[4.5, 1.3]], ["versicolor"]], [[[4.7, 1.6]], ["versicolor"]], [[[3.3, 1]], ["versicolor"]], [[[4.6, 1.3]], ["versicolor"]], [[[3.9, 1.4]], ["versicolor"]]]


# @pytest.mark.parametrize('nums, result', TEST_DTC_DATA)
# def test_knn_works(nums, result):
#     """Test knn works with flower data."""
#     from knn import KNN
#     knn = KNN(TRAIN_DTC_DATA)
#     assert knn.predict(nums) == result

a = [[1.4, 0.2, 5, 3.3], [1.5, 0.2, 5.3, 3.7], [1.4, 0.2, 4.6, 3.2], [4.3, 1.3, 6.2, 2.9], [3, 1.1, 5.1, 2.5], [4.1, 1.3, 5.7, 2.8]]

b = [[1.4, 0.2, 5.1, 3.5, 0], [1.4, 0.2, 4.9, 3, 0], [1.3, 0.2, 4.7, 3.2, 0], [4.7, 1.4, 7, 3.2, 1], [4.5, 1.5, 6.4, 3.2, 1], [4.9, 1.5, 6.9, 3.1, 1]]


def test_knn():
    """Docstring."""
    from knn import KNN
    knn = KNN(b)
    assert knn.predict(a) == [0, 0, 0, 1, 0, 1]
