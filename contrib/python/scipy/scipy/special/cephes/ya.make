PY23_LIBRARY()



NO_COMPILER_WARNINGS()


ADDINCL(
    contrib/python/scipy/scipy/special
)

PEERDIR(
    contrib/python/numpy
)

SRCS(
    airy.c
    bdtr.c
    beta.c
    btdtr.c
    cbrt.c
    chbevl.c
    chdtr.c
    const.c
    dawsn.c
    ellie.c
    ellik.c
    ellpe.c
    ellpj.c
    ellpk.c
    exp10.c
    exp2.c
    expn.c
    fdtr.c
    fresnl.c
    gamma.c
    gdtr.c
    gels.c
    hyp2f1.c
    hyperg.c
    i0.c
    i1.c
    igam.c
    igami.c
    incbet.c
    incbi.c
    j0.c
    j1.c
    jv.c
    k0.c
    k1.c
    kn.c
    kolmogorov.c
    lanczos.c
    mtherr.c
    nbdtr.c
    ndtr.c
    ndtri.c
    pdtr.c
    psi.c
    rgamma.c
    round.c
    scipy_iv.c
    shichi.c
    sici.c
    sincos.c
    sindg.c
    spence.c
    stdtr.c
    struve.c
    tandg.c
    tukey.c
    unity.c
    yn.c
    zeta.c
    zetac.c
)

END()
