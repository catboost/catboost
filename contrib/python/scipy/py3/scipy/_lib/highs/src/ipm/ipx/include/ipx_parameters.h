#ifndef IPX_PARAMETERS_H_
#define IPX_PARAMETERS_H_

#include "ipx_config.h"

#ifdef __cplusplus
extern "C" {
#endif
struct ipx_parameters {
    /* Solver control */
    ipxint display;
    const char* logfile;
    double print_interval;
    double time_limit;
    bool analyse_basis_data;

    /* Preprocessing */
    ipxint dualize;
    ipxint scale;

    /* Interior point method */
    ipxint ipm_maxiter;
    double ipm_feasibility_tol;
    double ipm_optimality_tol;
    double ipm_drop_primal;
    double ipm_drop_dual;

    /* Linear solver */
    double kkt_tol;

    /* Basis construction in IPM */
    ipxint crash_basis;
    double dependency_tol;
    double volume_tol;
    ipxint rows_per_slice;
    ipxint maxskip_updates;

    /* LU factorization */
    ipxint lu_kernel;
    double lu_pivottol;

    /* Crossover */
    ipxint crossover;
    double crossover_start;
    double pfeasibility_tol;
    double dfeasibility_tol;

    /* Debugging */
    ipxint debug;
    ipxint switchiter;
    ipxint stop_at_switch;
    ipxint update_heuristic;
    ipxint maxpasses;
};

#ifdef __cplusplus
}
#endif

#endif  /* IPX_PARAMETERS_H_ */
