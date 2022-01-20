PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    common_tests.py
    test_binned_statistic.py
    test_contingency.py
    test_continuous_basic.py
    test_discrete_basic.py
    test_distributions.py
    test_fit.py
    test_kdeoth.py
    test_morestats.py
    test_mstats_basic.py
    test_mstats_extras.py
    test_multivariate.py
    test_rank.py
    test_stats.py
    test_tukeylambda_stats.py
)

END()
