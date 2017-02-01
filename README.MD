[![Build Status](https://travis-ci.org/chamberi/data-structures.svg?branch=master)](https://travis-ci.org/chamberi/data-structures)
#Summary

Merge Sort orders a list so that it is ascending by dividing the list into progressively smaller partitions and then merging sorted combinations of those partitions.

msl : merge sort list

merge_sort() divides a list into progressively smaller partitions.

merge() compares two lists and re-arranges to form a new list from lowest to highest.

_random_list() returns a list of random numbers from 0 to 300 of random size less than 300. 


Timeit Information:


number of runs: 3
random merge_sort_timed: [0.9609010129997841, 0.9587301799992929, 0.9563081269998293]
average:  0.9586464399996354

number of runs: 3
best case merge_sort_timed: [0.6916009789993041, 0.6859150830005092, 0.6909817539999494]
average:  0.6894992719999209

number of runs: 3
reverse case merge_sort_timed: [0.7388693120001335, 0.7437362410000787, 0.7331475789997057]
average:  0.738584377333306


        test_merge_sort.py ............................

        ---------- coverage: platform darwin, python 2.7.12-final-0 ----------
        Name                      Stmts   Miss  Cover   Missing
        -------------------------------------------------------
        merge_sort.py                58     16    72%   76-97

        test_merge_sort.py ............................

        ---------- coverage: platform darwin, python 3.5.2-final-0 -----------
        Name                      Stmts   Miss  Cover   Missing
        -------------------------------------------------------
        merge_sort.py                58     16    72%   76-97
