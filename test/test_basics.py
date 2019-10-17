import doctest
from bucketstats import basics

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(basics))
    return tests

