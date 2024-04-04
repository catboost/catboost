#include "ipx_c.h"
#include "ipx_internal.h"
#include "lp_solver.h"

using namespace ipx;

struct ipx_parameters ipx_default_parameters() {
    ipx_parameters p;
    p.display = 1;
    p.logfile = nullptr;
    p.print_interval = 5.0;
    p.time_limit = -1.0;
    p.analyse_basis_data = false;
    p.dualize = -1;
    p.scale = 1;
    p.ipm_maxiter = 300;
    p.ipm_feasibility_tol = 1e-6;
    p.ipm_optimality_tol = 1e-8;
    p.ipm_drop_primal = 1e-9;
    p.ipm_drop_dual = 1e-9;
    p.kkt_tol = 0.3;
    p.crash_basis = 1;
    p.dependency_tol = 1e-6;
    p.volume_tol = 2.0;
    p.rows_per_slice = 10000;
    p.maxskip_updates = 10;
    p.lu_kernel = 0;
    p.lu_pivottol = 0.0625;
    p.crossover = 1;
    p.crossover_start = 1e-8;
    p.pfeasibility_tol = 1e-7;
    p.dfeasibility_tol = 1e-7;
    p.debug = 0;
    p.switchiter = -1;
    p.stop_at_switch = 0;
    p.update_heuristic = 1;
    p.maxpasses = -1;
    return p;
}

void ipx_new(void** p_self) {
    if (p_self) {
        try {
            LpSolver* solver = new LpSolver;
            *p_self = static_cast<void*>(solver);
        }
        catch (...) {
            *p_self = nullptr;
        }
    }
}

void ipx_free(void** p_self) {
    if (p_self && *p_self) {
        LpSolver* solver = static_cast<LpSolver*>(*p_self);
        delete solver;
        *p_self = nullptr;
    }
}

ipxint ipx_load_model(void* self, ipxint num_var, const double* obj,
                      const double* lb, const double* ub, ipxint num_constr,
                      const ipxint* Ap, const ipxint* Ai, const double* Ax,
                      const double* rhs, const char* constr_type) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->LoadModel(num_var, obj, lb, ub, num_constr, Ap, Ai, Ax, rhs,
                             constr_type);
}

ipxint ipx_load_ipm_starting_point(void* self, const double* x,
                                   const double* xl, const double* xu,
                                   const double* slack, const double* y,
                                   const double* zl, const double* zu) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->LoadIPMStartingPoint(x, xl, xu, slack, y, zl, zu);
}

ipxint ipx_solve(void* self) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->Solve();
}

struct ipx_info ipx_get_info(void* self) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetInfo();
}

ipxint ipx_get_interior_solution(void* self, double* x, double* xl, double* xu,
                                 double* slack, double* y, double* zl,
                                 double* zu) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetInteriorSolution(x, xl, xu, slack, y, zl, zu);
}

ipxint ipx_get_basic_solution(void* self, double* x, double* slack, double* y,
                              double* z, ipxint* cbasis, ipxint* vbasis) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetBasicSolution(x, slack, y, z, cbasis, vbasis);
}

struct ipx_parameters ipx_get_parameters(void* self) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetParameters();
}

void ipx_set_parameters(void* self, struct ipx_parameters new_parameters) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    solver->SetParameters(new_parameters);
}

void ipx_clear_model(void* self) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    solver->ClearModel();
}

ipxint ipx_get_basis(void* self, ipxint* cbasis, ipxint* vbasis) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetBasis(cbasis, vbasis);
}

ipxint ipx_get_iterate(void* self, double* x, double* y, double* zl,
                       double* zu, double* xl, double* xu) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetIterate(x, y, zl, zu, xl, xu);
}

ipxint ipx_get_kktmatrix(void* self, ipxint* AIp, ipxint* AIi, double* AIx,
                         double* g) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->GetKKTMatrix(AIp, AIi, AIx, g);
}

ipxint ipx_symbolic_invert(void* self, ipxint* rowcounts, ipxint* colcounts) {
    LpSolver* solver = static_cast<LpSolver*>(self);
    return solver->SymbolicInvert(rowcounts, colcounts);
}
