[![Build Status](https://travis-ci.org/CCallahanIV/data-structures.svg?branch=master)](https://travis-ci.org/CCallahanIV/data-structures)
#Summary

<<<<<<< HEAD
Insertion Sort orders a list so that it is ascending starting from one side and progressing to the other while finding the place of the next item by replacing it until it is larger than its next neighborand smaller than its last.

isl : insertion sort list

insertion_sort() orders a list so that it is ascending starting from one side and progressing to the other while finding the place of the next item by replacing it until it is larger than its next neighborand smaller than its last.

_random_list() returns a list of random numbers from 0 to 300 of size 150.

r is an unordered randomly filled list.
b is an ordered list of random numbers.
w is a reversed ordered list of random numbers.

Timeit Information:

number of runs: 3
random insertion_sort_timed: [0.03814769792370498, 0.04167536704335362, 0.03940913290716708]
average:  0.039744065958075225

number of runs: 3
best case insertion_sort_timed: [0.03972880798391998, 0.03789641603361815, 0.03818406199570745]
average:  0.03860309533774853

number of runs: 3
worst case insertion_sort_timed: [0.04407542699482292, 0.04105022503063083, 0.03949510189704597]
average:  0.04154025130749991


test_insertion_sort.py ............................

---------- coverage: platform darwin, python 3.5.2-final-0 -----------
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
insertion_sort.py            23      6    74%   44-50

test_insertion_sort.py ............................

---------- coverage: platform darwin, python 2.7.12-final-0 ----------
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
insertion_sort.py            23      6    74%   44-50

