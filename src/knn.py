"""Module for k nearest neighbor."""
import numpy as np


# K NEAREST NEIGHBOR (KNN)
#
# CodeFellows 401d5
# Submission Date:
#
# Authors:  Colin Lamont <https://github.com/chamberi>
#           Ben Shields <https://github.com/iamrobinhood12345>
#
# URL:


class KNN(object):
    """K Nearest Neighbor Classifier Object."""

    """clf.predict(self, data): returns labels for your test data."""

    def __init__(self, data, k=5):
        """Initialize the KNN object."""
        self.data = data
        if k > len(data) or type(k) is not int or k <= 0:
            raise ValueError("The number of neighbors to predict against must be an integer greater than 0 and less than size of data.")
        self.k = 5

    def predict(self, evals, k=None):
        """Predict function that compares distances."""
        if not k:
            k = self.k
        win = []
        for eval_item in evals:        # get each set of eval
            distance_list = []
            for data_idx in range(len(self.data)):
                squares_sum = 0.0
                for item_idx in range(len(eval_item)):
                    squares_sum += (eval_item[item_idx] - self.data[data_idx][item_idx]) ** 2
                distance_list.append((np.sqrt(squares_sum), self.data[data_idx][-1]))
            distance_list.sort()
            win.append(self._get_winner(distance_list, k))
        return win

    def _get_winner(self, distance_list, k):
        """Generate the top classes and the majority closest class."""
        classes, winner = [], []
        counter = {}
        classes = [instance[1] for instance in distance_list[:k]]
        unique_classes = sorted(set(classes), reverse=True)
        for i in unique_classes:
            counter.setdefault(i, len(list(x for x in classes if x == i)))
        winner = sorted([(value, key) for (key, value) in counter.items()], reverse=True)
        if winner[0][0] == winner[1][0]:
            self._get_winner(distance_list, k=k - 1)
        return winner[0][1]
