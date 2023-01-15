PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    test_orthogonal_eval.py
    test_mpmath.py
    test_spfun_stats.py
    test_spherical_bessel.py
    test_cdflib.py
    test_sici.py
    test_boxcox.py
    test_loggamma.py
    test_digamma.py
    test_data.py
    test_precompute_expn_asy.py
    test_logit.py
    test_basic.py
    test_spence.py
    test_lambertw.py
    test_ellip_harm.py
    test_cython_special.py
    test_precompute_gammainc.py
    test_orthogonal.py
    test_gammainc.py
    test_precompute_utils.py
)

END()
