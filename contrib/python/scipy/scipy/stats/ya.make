PY23_LIBRARY()

LICENSE(BSD-3-Clause)



NO_COMPILER_WARNINGS()

NO_LINT()

PY_SRCS(
    NAMESPACE scipy.stats

    __init__.py
    _distr_params.py
    kde.py
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
    # mvn DISABLED due to f2c compilation problems https://mail.scipy.org/pipermail/scipy-dev/2010-January/013713.html
    #mvndst.f
    #mvn-f2pywrappers.f
    #mvnmodule.c
    statlibmodule.c
)

PY_REGISTER(scipy.stats.statlib)
#PY_REGISTER(scipy.stats.mvn)  DISABLED

PEERDIR(
    contrib/python/scipy/scipy/stats/statlib
    contrib/python/numpy
)

END()
