PY2TEST()



VERSION(1.2.3)

ORIGINAL_SOURCE(mirror://pypi/s/scipy/scipy-1.2.3.tar.gz)

SIZE(MEDIUM)

FORK_TESTS()

PEERDIR(
    contrib/python/scipy/py2
    contrib/python/scipy/py2/scipy/conftest
)

NO_LINT()

NO_CHECK_IMPORTS()

TEST_SRCS(
    __init__.py
    common_tests.py
    test_binned_statistic.py
    test_contingency.py
    test_continuous_basic.py
    test_discrete_basic.py
    test_discrete_distns.py
    test_fit.py
    test_morestats.py
    test_mstats_basic.py
    test_mstats_extras.py
    test_rank.py
    test_stats.py
    test_tukeylambda_stats.py
)

END()
