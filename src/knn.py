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
            raise ValueError("The number of neighbors to predict against must be a integer greater than 0 and less than size of data.")
        self.k = 5

    def predict(self, evals, k=None):
        """Predict function that compares distances."""
        import pdb; pdb.set_trace()
        if not k:
            k = self.k
        for eval_item in evals:        # get each set of eval
            distance_list, classes, winner = [], [], []
            counter = {}
            for data_idx in range(len(self.data)):# get each set of data
                squares_sum = 0.0
                for item_idx in range(len(eval_item)):    # get each item in eval
                    squares_sum += (eval_item[item_idx] - self.data[data_idx][item_idx]) ** 2  # compare eval item to each item in data
                distance_list.append((np.sqrt(squares_sum), self.data[data_idx][-1]))   # append distance and class
            distance_list.sort()                                                    # sort by shortest distances
            classes = [instance[1] for instance in distance_list[:k]]               # get list of classes
            unique_classes = sorted(set(classes), reverse=True)                       # unique set of classes
            for i in unique_classes:
                counter.setdefault(i, len(list(x for x in classes if x == i)))      # ? append number of classes and class to counter
            winner = sorted([(value, key) for (key, value) in counter.items()], reverse=True)
            if winner[0][0] == winner[1][0]:
                self.predict(evals, k=k - 1)                                            # if first 
            else:
                return winner[0][1]
