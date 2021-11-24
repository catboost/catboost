PY23_LIBRARY()

LICENSE(BSD-3-Clause)



VERSION(0.18.1)

PEERDIR(
    contrib/python/numpy
    contrib/python/scipy/scipy/spatial
    contrib/python/scipy/scipy/fftpack
    contrib/python/scipy/scipy/signal
    contrib/python/scipy/scipy/ndimage
    contrib/python/scipy/scipy/stats
    contrib/python/scipy/scipy/constants
    contrib/python/scipy/scipy/cluster
    contrib/python/scipy/scipy/odr
    contrib/python/scipy/scipy/_lib
    contrib/python/scipy/scipy/linalg
    contrib/python/scipy/scipy/sparse
)

IF (OS_WINDOWS)
    CFLAGS(-D_USE_MATH_DEFINES)
ENDIF()

ADDINCL(
    FOR cython contrib/python/scipy
    contrib/python/scipy/scipy/interpolate
    contrib/python/scipy/scipy/io/matlab
    contrib/python/scipy/scipy/special
    contrib/python/scipy/scipy/special/c_misc
)

NO_LINT()
NO_COMPILER_WARNINGS()

SRCS(
    scipy/integrate/_dopmodule.c
    scipy/integrate/_odepackmodule.c
    scipy/integrate/_quadpackmodule.c
    scipy/integrate/lsodamodule.c
    scipy/integrate/vodemodule.c

    scipy/integrate/dop/dop853.f
    scipy/integrate/dop/dopri5.f

    # scipy/integrate/mach/d1mach.f in scipy/special/mach/
    # scipy/integrate/mach/xerror.f in scipy/special/mach/

    scipy/integrate/odepack/blkdta000.f
    scipy/integrate/odepack/bnorm.f
    scipy/integrate/odepack/cfode.f
    scipy/integrate/odepack/ewset.f
    scipy/integrate/odepack/fnorm.f
    scipy/integrate/odepack/intdy.f
    scipy/integrate/odepack/lsoda.f
    scipy/integrate/odepack/prja.f
    scipy/integrate/odepack/solsy.f
    scipy/integrate/odepack/srcma.f
    scipy/integrate/odepack/stoda.f
    scipy/integrate/odepack/vmnorm.f
    scipy/integrate/odepack/vode.f
    scipy/integrate/odepack/xerrwv.f
    # scipy/integrate/odepack/xsetf.f in vode.f
    # scipy/integrate/odepack/xsetun.f in vode.f
    scipy/integrate/odepack/zvode.f

    scipy/integrate/quadpack/dqag.f
    scipy/integrate/quadpack/dqagi.f
    scipy/integrate/quadpack/dqagie.f
    scipy/integrate/quadpack/dqagp.f
    scipy/integrate/quadpack/dqagpe.f
    scipy/integrate/quadpack/dqags.f
    scipy/integrate/quadpack/dqagse.f
    scipy/integrate/quadpack/dqawc.f
    scipy/integrate/quadpack/dqawce.f
    scipy/integrate/quadpack/dqawf.f
    scipy/integrate/quadpack/dqawfe.f
    scipy/integrate/quadpack/dqawo.f
    scipy/integrate/quadpack/dqawoe.f
    scipy/integrate/quadpack/dqaws.f
    scipy/integrate/quadpack/dqawse.f
    scipy/integrate/quadpack/dqc25c.f
    scipy/integrate/quadpack/dqc25f.f
    scipy/integrate/quadpack/dqc25s.f
    scipy/integrate/quadpack/dqcheb.f
    scipy/integrate/quadpack/dqelg.f
    scipy/integrate/quadpack/dqk15.f
    scipy/integrate/quadpack/dqk15i.f
    scipy/integrate/quadpack/dqk15w.f
    scipy/integrate/quadpack/dqk21.f
    scipy/integrate/quadpack/dqk31.f
    scipy/integrate/quadpack/dqk41.f
    scipy/integrate/quadpack/dqk51.f
    scipy/integrate/quadpack/dqk61.f
    scipy/integrate/quadpack/dqmomo.f
    scipy/integrate/quadpack/dqng.f
    scipy/integrate/quadpack/dqpsrt.f
    scipy/integrate/quadpack/dqwgtc.f
    scipy/integrate/quadpack/dqwgtf.f
    scipy/integrate/quadpack/dqwgts.f

    scipy/interpolate/fitpack/bispeu.f
    scipy/interpolate/fitpack/bispev.f
    scipy/interpolate/fitpack/clocur.f
    scipy/interpolate/fitpack/cocosp.f
    scipy/interpolate/fitpack/concon.f
    scipy/interpolate/fitpack/concur.f
    scipy/interpolate/fitpack/cualde.f
    scipy/interpolate/fitpack/curev.f
    scipy/interpolate/fitpack/curfit.f
    scipy/interpolate/fitpack/dblint.f
    scipy/interpolate/fitpack/evapol.f
    scipy/interpolate/fitpack/fourco.f
    scipy/interpolate/fitpack/fpader.f
    scipy/interpolate/fitpack/fpadno.f
    scipy/interpolate/fitpack/fpadpo.f
    scipy/interpolate/fitpack/fpback.f
    scipy/interpolate/fitpack/fpbacp.f
    scipy/interpolate/fitpack/fpbfout.f
    scipy/interpolate/fitpack/fpbisp.f
    scipy/interpolate/fitpack/fpbspl.f
    scipy/interpolate/fitpack/fpchec.f
    scipy/interpolate/fitpack/fpched.f
    scipy/interpolate/fitpack/fpchep.f
    scipy/interpolate/fitpack/fpclos.f
    scipy/interpolate/fitpack/fpcoco.f
    scipy/interpolate/fitpack/fpcons.f
    scipy/interpolate/fitpack/fpcosp.f
    scipy/interpolate/fitpack/fpcsin.f
    scipy/interpolate/fitpack/fpcurf.f
    scipy/interpolate/fitpack/fpcuro.f
    scipy/interpolate/fitpack/fpcyt1.f
    scipy/interpolate/fitpack/fpcyt2.f
    scipy/interpolate/fitpack/fpdeno.f
    scipy/interpolate/fitpack/fpdisc.f
    scipy/interpolate/fitpack/fpfrno.f
    scipy/interpolate/fitpack/fpgivs.f
    scipy/interpolate/fitpack/fpgrdi.f
    scipy/interpolate/fitpack/fpgrpa.f
    scipy/interpolate/fitpack/fpgrre.f
    scipy/interpolate/fitpack/fpgrsp.f
    scipy/interpolate/fitpack/fpinst.f
    scipy/interpolate/fitpack/fpintb.f
    scipy/interpolate/fitpack/fpknot.f
    scipy/interpolate/fitpack/fpopdi.f
    scipy/interpolate/fitpack/fpopsp.f
    scipy/interpolate/fitpack/fporde.f
    scipy/interpolate/fitpack/fppara.f
    scipy/interpolate/fitpack/fppasu.f
    scipy/interpolate/fitpack/fpperi.f
    scipy/interpolate/fitpack/fppocu.f
    scipy/interpolate/fitpack/fppogr.f
    scipy/interpolate/fitpack/fppola.f
    scipy/interpolate/fitpack/fprank.f
    scipy/interpolate/fitpack/fprati.f
    scipy/interpolate/fitpack/fpregr.f
    scipy/interpolate/fitpack/fprota.f
    scipy/interpolate/fitpack/fprppo.f
    scipy/interpolate/fitpack/fprpsp.f
    scipy/interpolate/fitpack/fpseno.f
    scipy/interpolate/fitpack/fpspgr.f
    scipy/interpolate/fitpack/fpsphe.f
    scipy/interpolate/fitpack/fpsuev.f
    scipy/interpolate/fitpack/fpsurf.f
    scipy/interpolate/fitpack/fpsysy.f
    scipy/interpolate/fitpack/fptrnp.f
    scipy/interpolate/fitpack/fptrpe.f
    scipy/interpolate/fitpack/insert.f
    scipy/interpolate/fitpack/parcur.f
    scipy/interpolate/fitpack/parder.f
    scipy/interpolate/fitpack/pardeu.f
    scipy/interpolate/fitpack/parsur.f
    scipy/interpolate/fitpack/percur.f
    scipy/interpolate/fitpack/pogrid.f
    scipy/interpolate/fitpack/polar.f
    scipy/interpolate/fitpack/profil.f
    scipy/interpolate/fitpack/regrid.f
    scipy/interpolate/fitpack/spalde.f
    scipy/interpolate/fitpack/spgrid.f
    scipy/interpolate/fitpack/sphere.f
    scipy/interpolate/fitpack/splder.f
    scipy/interpolate/fitpack/splev.f
    scipy/interpolate/fitpack/splint.f
    scipy/interpolate/fitpack/sproot.f
    scipy/interpolate/fitpack/surev.f
    scipy/interpolate/fitpack/surfit.f

    scipy/interpolate/src/_fitpackmodule.c
    scipy/interpolate/src/_interpolate.cpp
    scipy/interpolate/src/dfitpackmodule.c
    scipy/interpolate/src/dfitpack-f2pywrappers.f

    scipy/optimize/_minpackmodule.c
    scipy/optimize/zeros.c

    scipy/optimize/cobyla/cobyla2.f
    scipy/optimize/cobyla/_cobylamodule.c
    scipy/optimize/cobyla/trstlp.f

    scipy/optimize/lbfgsb/lbfgsb.f
    scipy/optimize/lbfgsb/linpack.f
    scipy/optimize/lbfgsb/timer.f
    scipy/optimize/lbfgsb/_lbfgsbmodule.c

    scipy/optimize/minpack/chkder.f
    scipy/optimize/minpack/dogleg.f
    scipy/optimize/minpack/dpmpar.f
    scipy/optimize/minpack/enorm.f
    scipy/optimize/minpack/fdjac1.f
    scipy/optimize/minpack/fdjac2.f
    scipy/optimize/minpack/hybrd1.f
    scipy/optimize/minpack/hybrd.f
    scipy/optimize/minpack/hybrj1.f
    scipy/optimize/minpack/hybrj.f
    scipy/optimize/minpack/lmder1.f
    scipy/optimize/minpack/lmder.f
    scipy/optimize/minpack/lmdif1.f
    scipy/optimize/minpack/lmdif.f
    scipy/optimize/minpack/lmpar.f
    scipy/optimize/minpack/lmstr1.f
    scipy/optimize/minpack/lmstr.f
    scipy/optimize/minpack/qform.f
    scipy/optimize/minpack/qrfac.f
    scipy/optimize/minpack/qrsolv.f
    scipy/optimize/minpack/r1mpyq.f
    scipy/optimize/minpack/r1updt.f
    scipy/optimize/minpack/rwupdt.f

    scipy/optimize/minpack2/minpack2module.c
    scipy/optimize/minpack2/dcsrch.f
    scipy/optimize/minpack2/dcstep.f

    scipy/optimize/nnls/nnls.f
    scipy/optimize/nnls/_nnlsmodule.c

    scipy/optimize/slsqp/_slsqpmodule.c
    scipy/optimize/slsqp/slsqp_optmz.f

    scipy/optimize/tnc/moduleTNC.c
    scipy/optimize/tnc/tnc.c

    scipy/optimize/Zeros/bisect.c
    scipy/optimize/Zeros/brenth.c
    scipy/optimize/Zeros/brentq.c
    scipy/optimize/Zeros/ridder.c

    scipy/special/Faddeeva.cc
    scipy/special/_faddeeva.cxx
    scipy/special/_logit.c
    scipy/special/amos_wrappers.c
    scipy/special/cdf_wrappers.c
    scipy/special/sf_error.c

    scipy/special/amos/dgamln.f
    scipy/special/amos/dsclmr.f
    scipy/special/amos/fdump.f
    scipy/special/amos/zabs.f
    scipy/special/amos/zacai.f
    scipy/special/amos/zacon.f
    scipy/special/amos/zairy.f
    scipy/special/amos/zasyi.f
    scipy/special/amos/zbesh.f
    scipy/special/amos/zbesi.f
    scipy/special/amos/zbesj.f
    scipy/special/amos/zbesk.f
    scipy/special/amos/zbesy.f
    scipy/special/amos/zbinu.f
    scipy/special/amos/zbiry.f
    scipy/special/amos/zbknu.f
    scipy/special/amos/zbuni.f
    scipy/special/amos/zbunk.f
    scipy/special/amos/zdiv.f
    scipy/special/amos/zexp.f
    scipy/special/amos/zkscl.f
    scipy/special/amos/zlog.f
    scipy/special/amos/zmlri.f
    scipy/special/amos/zmlt.f
    scipy/special/amos/zrati.f
    scipy/special/amos/zs1s2.f
    scipy/special/amos/zseri.f
    scipy/special/amos/zshch.f
    scipy/special/amos/zsqrt.f
    scipy/special/amos/zuchk.f
    scipy/special/amos/zunhj.f
    scipy/special/amos/zuni1.f
    scipy/special/amos/zuni2.f
    scipy/special/amos/zunik.f
    scipy/special/amos/zunk1.f
    scipy/special/amos/zunk2.f
    scipy/special/amos/zuoik.f
    scipy/special/amos/zwrsk.f

    scipy/special/c_misc/besselpoly.c
    scipy/special/c_misc/double2.h
    scipy/special/c_misc/fsolve.c
    scipy/special/c_misc/gammaincinv.c
    scipy/special/c_misc/gammasgn.c
    scipy/special/c_misc/misc.h
    scipy/special/c_misc/poch.c
    scipy/special/c_misc/struve.c

    scipy/special/cdflib/algdiv.f
    scipy/special/cdflib/alngam.f
    scipy/special/cdflib/alnrel.f
    scipy/special/cdflib/apser.f
    scipy/special/cdflib/basym.f
    scipy/special/cdflib/bcorr.f
    scipy/special/cdflib/betaln.f
    scipy/special/cdflib/bfrac.f
    scipy/special/cdflib/bgrat.f
    scipy/special/cdflib/bpser.f
    scipy/special/cdflib/bratio.f
    scipy/special/cdflib/brcmp1.f
    scipy/special/cdflib/brcomp.f
    scipy/special/cdflib/bup.f
    scipy/special/cdflib/cdfbet.f
    scipy/special/cdflib/cdfbin.f
    scipy/special/cdflib/cdfchi.f
    scipy/special/cdflib/cdfchn.f
    scipy/special/cdflib/cdff.f
    scipy/special/cdflib/cdffnc.f
    scipy/special/cdflib/cdfgam.f
    scipy/special/cdflib/cdfnbn.f
    scipy/special/cdflib/cdfnor.f
    scipy/special/cdflib/cdfpoi.f
    scipy/special/cdflib/cdft.f
    scipy/special/cdflib/cdftnc.f
    scipy/special/cdflib/cumbet.f
    scipy/special/cdflib/cumbin.f
    scipy/special/cdflib/cumchi.f
    scipy/special/cdflib/cumchn.f
    scipy/special/cdflib/cumf.f
    scipy/special/cdflib/cumfnc.f
    scipy/special/cdflib/cumgam.f
    scipy/special/cdflib/cumnbn.f
    scipy/special/cdflib/cumnor.f
    scipy/special/cdflib/cumpoi.f
    scipy/special/cdflib/cumt.f
    scipy/special/cdflib/cumtnc.f
    scipy/special/cdflib/devlpl.f
    scipy/special/cdflib/dinvnr.f
    scipy/special/cdflib/dinvr.f
    scipy/special/cdflib/dt1.f
    scipy/special/cdflib/dzror.f
    # scipy/special/cdflib/erf.f in libf2c
    scipy/special/cdflib/erfc1.f
    scipy/special/cdflib/esum.f
    scipy/special/cdflib/exparg.f
    scipy/special/cdflib/fpser.f
    scipy/special/cdflib/gam1.f
    scipy/special/cdflib/gaminv.f
    scipy/special/cdflib/gamln.f
    scipy/special/cdflib/gamln1.f
    scipy/special/cdflib/gamma_fort.f
    scipy/special/cdflib/grat1.f
    scipy/special/cdflib/gratio.f
    scipy/special/cdflib/gsumln.f
    scipy/special/cdflib/ipmpar.f
    scipy/special/cdflib/psi_fort.f
    scipy/special/cdflib/rcomp.f
    scipy/special/cdflib/rexp.f
    scipy/special/cdflib/rlog.f
    scipy/special/cdflib/rlog1.f
    scipy/special/cdflib/spmpar.f
    scipy/special/cdflib/stvaln.f

    scipy/special/cephes/airy.c
    scipy/special/cephes/bdtr.c
    scipy/special/cephes/beta.c
    scipy/special/cephes/btdtr.c
    scipy/special/cephes/cbrt.c
    scipy/special/cephes/chbevl.c
    scipy/special/cephes/chdtr.c
    scipy/special/cephes/const.c
    scipy/special/cephes/dawsn.c
    scipy/special/cephes/ellie.c
    scipy/special/cephes/ellik.c
    scipy/special/cephes/ellpe.c
    scipy/special/cephes/ellpj.c
    scipy/special/cephes/ellpk.c
    scipy/special/cephes/exp10.c
    scipy/special/cephes/exp2.c
    scipy/special/cephes/expn.c
    scipy/special/cephes/fdtr.c
    scipy/special/cephes/fresnl.c
    scipy/special/cephes/gamma.c
    scipy/special/cephes/gdtr.c
    scipy/special/cephes/gels.c
    scipy/special/cephes/hyp2f1.c
    scipy/special/cephes/hyperg.c
    scipy/special/cephes/i0.c
    scipy/special/cephes/i1.c
    scipy/special/cephes/igam.c
    scipy/special/cephes/igami.c
    scipy/special/cephes/incbet.c
    scipy/special/cephes/incbi.c
    scipy/special/cephes/j0.c
    scipy/special/cephes/j1.c
    scipy/special/cephes/jv.c
    scipy/special/cephes/k0.c
    scipy/special/cephes/k1.c
    scipy/special/cephes/kn.c
    scipy/special/cephes/kolmogorov.c
    scipy/special/cephes/lanczos.c
    scipy/special/cephes/mtherr.c
    scipy/special/cephes/nbdtr.c
    scipy/special/cephes/ndtr.c
    scipy/special/cephes/ndtri.c
    scipy/special/cephes/pdtr.c
    scipy/special/cephes/psi.c
    scipy/special/cephes/rgamma.c
    scipy/special/cephes/round.c
    scipy/special/cephes/scipy_iv.c
    scipy/special/cephes/shichi.c
    scipy/special/cephes/sici.c
    scipy/special/cephes/sincos.c
    scipy/special/cephes/sindg.c
    scipy/special/cephes/spence.c
    scipy/special/cephes/stdtr.c
    scipy/special/cephes/struve.c
    scipy/special/cephes/tandg.c
    scipy/special/cephes/tukey.c
    scipy/special/cephes/unity.c
    scipy/special/cephes/yn.c
    scipy/special/cephes/zeta.c
    scipy/special/cephes/zetac.c

    scipy/special/mach/d1mach.f
    scipy/special/mach/i1mach.f
    scipy/special/mach/xerror.f

    scipy/special/specfun/specfun.f
    scipy/special/specfun_wrappers.c
    scipy/special/specfunmodule.c
)



PY_SRCS(
    TOP_LEVEL

    scipy/__init__.py
    scipy/__config__.py
    scipy/version.py

    scipy/integrate/__init__.py
    scipy/integrate/quadpack.py
    scipy/integrate/quadrature.py
    scipy/integrate/_ode.py
    scipy/integrate/odepack.py
    scipy/integrate/_bvp.py

    scipy/interpolate/__init__.py
    scipy/interpolate/_cubic.py
    scipy/interpolate/fitpack2.py
    scipy/interpolate/fitpack.py
    scipy/interpolate/interpolate.py
    scipy/interpolate/interpolate_wrapper.py
    scipy/interpolate/ndgriddata.py
    scipy/interpolate/polyint.py
    scipy/interpolate/rbf.py

    CYTHON_C
    scipy/interpolate/_ppoly.pyx
    scipy/interpolate/interpnd.pyx

    scipy/io/__init__.py
    scipy/io/_fortran.py
    scipy/io/idl.py
    scipy/io/mmio.py
    scipy/io/netcdf.py
    scipy/io/wavfile.py

    scipy/io/arff/__init__.py
    scipy/io/arff/arffread.py

    scipy/io/harwell_boeing/__init__.py
    scipy/io/harwell_boeing/_fortran_format_parser.py
    scipy/io/harwell_boeing/hb.py

    scipy/io/matlab/__init__.py
    scipy/io/matlab/byteordercodes.py
    scipy/io/matlab/mio4.py
    scipy/io/matlab/mio5_params.py
    scipy/io/matlab/mio5.py
    scipy/io/matlab/miobase.py
    scipy/io/matlab/mio.py

    CYTHON_C
    scipy/io/matlab/mio5_utils.pyx
    scipy/io/matlab/mio_utils.pyx
    scipy/io/matlab/streams.pyx

    scipy/optimize/__init__.py
    scipy/optimize/_basinhopping.py
    scipy/optimize/cobyla.py
    scipy/optimize/_differentialevolution.py
    scipy/optimize/_hungarian.py
    scipy/optimize/lbfgsb.py
    scipy/optimize/linesearch.py
    scipy/optimize/_linprog.py
    scipy/optimize/_lsq/bvls.py
    scipy/optimize/_lsq/common.py
    scipy/optimize/_lsq/dogbox.py
    scipy/optimize/_lsq/__init__.py
    scipy/optimize/_lsq/least_squares.py
    scipy/optimize/_lsq/lsq_linear.py
    scipy/optimize/_lsq/trf_linear.py
    scipy/optimize/_lsq/trf.py
    scipy/optimize/_minimize.py
    scipy/optimize/minpack.py
    scipy/optimize/nnls.py
    scipy/optimize/nonlin.py
    scipy/optimize/_numdiff.py
    scipy/optimize/optimize.py
    scipy/optimize/_root.py
    scipy/optimize/slsqp.py
    scipy/optimize/_spectral.py
    scipy/optimize/tnc.py
    scipy/optimize/_trustregion_dogleg.py
    scipy/optimize/_trustregion_ncg.py
    scipy/optimize/_trustregion.py
    scipy/optimize/_tstutils.py
    scipy/optimize/zeros.py

    CYTHON_CPP
    scipy/optimize/_lsq/givens_elimination.pyx

    CYTHON_C
    scipy/optimize/_group_columns.pyx

    scipy/special/__init__.py
    scipy/special/lambertw.py
    scipy/special/orthogonal.py
    scipy/special/_ellip_harm.py
    scipy/special/basic.py
    scipy/special/spfun_stats.py
#   scipy/special/_testutils.py
#   scipy/special/_mptestutils.py
    scipy/special/add_newdocs.py
    scipy/special/_spherical_bessel.py

    scipy/special/_precompute/__init__.py
    scipy/special/_precompute/expn_asy.py
    scipy/special/_precompute/gammainc_asy.py
    scipy/special/_precompute/utils.py

    CYTHON_C
    scipy/special/_comb.pyx
    scipy/special/_ufuncs.pyx
    scipy/special/_ellip_harm_2.pyx

    CYTHON_CPP
    scipy/special/_ufuncs_cxx.pyx
)

PY_REGISTER(
    scipy.integrate._odepack
    scipy.integrate._quadpack
    scipy.integrate.vode
    scipy.integrate._dop
    scipy.integrate.lsoda

    scipy.interpolate._fitpack
    scipy.interpolate.dfitpack
    scipy.interpolate._interpolate

    scipy.optimize._cobyla
    scipy.optimize._lbfgsb
    scipy.optimize._minpack
    scipy.optimize._nnls
    scipy.optimize._slsqp
    scipy.optimize._zeros
    scipy.optimize.minpack2
    scipy.optimize.moduleTNC

    scipy.special.specfun
)

END()

RECURSE_FOR_TESTS(
    scipy/_build_utils/tests
    scipy/cluster/tests
    scipy/constants/tests
    scipy/fftpack/tests
    scipy/integrate/tests
    scipy/interpolate/tests
    scipy/io/arff/tests
    scipy/io/harwell_boeing/tests
    scipy/io/matlab/tests
    scipy/io/tests
    scipy/_lib/tests
    scipy/linalg/tests
    scipy/misc/tests
    scipy/ndimage/tests
    scipy/odr/tests
    scipy/optimize/tests
    scipy/signal/tests
    scipy/sparse/csgraph/tests
    scipy/sparse/linalg/dsolve/tests
    scipy/sparse/linalg/eigen/arpack/tests
    scipy/sparse/linalg/eigen/lobpcg/tests
    scipy/sparse/linalg/isolve/tests
    scipy/sparse/linalg/tests
    scipy/sparse/tests
    scipy/spatial/tests
    scipy/special/tests
    scipy/stats/tests
)
