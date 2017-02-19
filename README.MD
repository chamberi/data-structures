[![Build Status](https://travis-ci.org/chamberi/data-structures.svg?branch=master)](https://travis-ci.org/chamberi/data-structures)
#Summary
        
        ## K-Nearest Neighbors

        predict(): predicts using the neares
        _get_winner(): finds the k closest classes, and returns the closest one to the predict function.

        ipython
        import numpy as np
        run knn.py
        a = [[1.4, 0.2, 5, 3.3], [1.5, 0.2, 5.3, 3.7], [1.4, 0.2, 4.6, 3.2], [4.3, 1.3, 6.2, 2.9], [4, 1.2, 5.8, 2.6], [4.1, 1.3, 5.7, 2.8]]
        b = [[1.4, 0.2, 5.1, 3.5, 0], [1.4, 0.2, 4.9, 3, 0], [1.3, 0.2, 4.7, 3.2, 0], [4.7, 1.4, 7, 3.2, 1], [4.5, 1.5, 6.4, 3.2, 1], [4.9, 1.5, 6.9, 3.1, 1]]
        z = KNN(b)
        z.predict(a) == [0, 0, 0, 1, 1, 1]


## Tests and Coverage

                test_knn.py .....

                ---------- coverage: platform darwin, python 3.5.2-final-0 -----------
                Name                      Stmts   Miss  Cover   Missing
                -------------------------------------------------------
                knn.py                       32      1    97%   54
