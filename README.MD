[![Build Status](https://travis-ci.org/CCallahanIV/data-structures.svg?branch=master)](https://travis-ci.org/CCallahanIV/data-structures)
#Summary

        insert(self, string): will insert the input string into the trie. If character in the input string is already present, it will be ignored.
        contains(self, string): will return True if the string is in the trie, False if not.
        size(self): will return the total number of words contained within the trie. 0 if empty.
        remove(self, string): will remove the given string from the trie. If the word does not exist, will raise an appropriate exception


# Coverage:

test_trie.py ...........

---------- coverage: platform darwin, python 2.7.12-final-0 ----------
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
trie.py                      67      6    91%   92, 98-99, 101-103

test_trie.py ...........

---------- coverage: platform darwin, python 3.5.2-final-0 -----------
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
trie.py                      67      6    91%   92, 98-99, 101-103