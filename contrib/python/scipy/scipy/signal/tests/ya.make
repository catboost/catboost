PROGRAM()



NO_LINT()
NO_CHECK_IMPORTS()

PEERDIR(
    contrib/python/nose/runner

    contrib/python/scipy
)

TEST_SRCS(
    mpsig.py
    test_array_tools.py
    test_cont2discrete.py
    test_dltisys.py
    test_filter_design.py
    test_fir_filter_design.py
    test_ltisys.py
    test_max_len_seq.py
    test_peak_finding.py
    test_savitzky_golay.py
    test_signaltools.py
    test_spectral.py
    test_upfirdn.py
    test_waveforms.py
    test_wavelets.py
    test_windows.py
)

END()
