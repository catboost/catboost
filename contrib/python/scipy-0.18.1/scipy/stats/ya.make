LIBRARY()



NO_COMPILER_WARNINGS()

PY_SRCS(
    NAMESPACE scipy.stats

    __init__.py
    _distr_params.py
    #    kde.py  DISABLED due to f2c compilation problems https://mail.scipy.org/pipermail/scipy-dev/2010-January/013713.html
    mstats.py
    distributions.py
    vonmises.py
    _continuous_distns.py
    contingency.py
    _stats_mstats_common.py
    _multivariate.py
    mstats_extras.py
    morestats.py
    _tukeylambda_stats.py
    _distn_infrastructure.py
    mstats_basic.py
    _constants.py
    stats.py
    _binned_statistic.py
    _discrete_distns.py

    _stats.pyx
)

SRCS(
    #    mvndst.f  DISABLED
    #    mvn-f2pywrappers.f  DISABLED
    #    mvnmodule.c  DISABLED
    statlibmodule.c
)

PY_REGISTER(scipy.stats.statlib)
#PY_REGISTER(scipy.stats.mvn)  DISABLED

PEERDIR(
    contrib/python/numpy/numpy/f2py/src

    contrib/python/scipy-0.18.1/scipy/stats/statlib
)

END()
