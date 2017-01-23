#Summary

    insert(self, val): will insert the value val into the BST. If val is already present, it will be ignored.
    search(self, val): will return the node containing that value, else None.
    size(self): will return the integer size of the BST (equal to the total number of values stored in the tree). It will return 0 if the tree is empty.
    depth(self): will return an integer representing the total number of levels in the tree. If there is one value, the depth should be 1, if two values it will be 2, if three values it may be 2 or three, depending, etc.
    contains(self, val): will return True if val is in the BST, False if not.
    balance(self): will return an integer, positive or negative that represents how well balanced the tree is. Trees which are higher on the left than the right should return a positive value, trees which are higher on the right than the left should return a negative value. An ideally-balanced tree should return 0.


# Coverage:

15 of 15 Passing Tests:

---------- coverage: platform darwin, python 3.5.2-final-0 -----------

Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/bst.py                       65     10    85%   30-34, 64, 83-85, 115


---------- coverage: platform darwin, python 2.7.12-final-0 ----------

Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/bst.py                       65     10    85%   30-34, 64, 83-85, 115