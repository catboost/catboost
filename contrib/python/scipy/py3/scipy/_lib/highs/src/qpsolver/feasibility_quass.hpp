#ifndef __SRC_LIB_FEASIBILITYQUASS_HPP__
#define __SRC_LIB_FEASIBILITYQUASS_HPP__

#include "crashsolution.hpp"
#include "basis.hpp"
#include "runtime.hpp"

void computestartingpoint_quass(Runtime& runtime, CrashSolution& result) {
  /*
   creates and solves the feasibility problem
   min 
    gamma
   s.t. 
    b - gamma <= Ax <= b + gamma
    l - gamma <=  x <= u + gamma
        gamma >= 0

    with initial values:
      compute x = l or b if possible, x = 0 for free variables
      compute Ax = ra
      compute initial gamma (large enough to make above system feasible)
      initialactive: constraint at which gamma attains maximum, all x for bounded x
      initialinactive: all x for free x


    FTRAN: B lambda = c (== 1, 00000)
    PRICE (find most negative/positive lambda not corresponding to gamma)
    RATIOTEST
    UPDATE (BTRAN)
  */
 //Basis basis();









  // // create artificial bounds for free variables
  // if (runtime.settings.phase1boundfreevars) {
  //   for (HighsInt i=0; i<runtime.instance.num_var; i++) {
  //     if (isfreevar(runtime, i)) {
  //       lp.col_lower_[i] = -1E5;
  //       lp.col_upper_[i] = 1E5;
  //     }
  //   }
  // }

  // if (runtime.settings.phase1movefreevarsbasic) {

  // }

  // runtime.statistics.phase1_iterations = 0;

  // HighsModelStatus phase1stat = highs.getModelStatus();
  // if (phase1stat == HighsModelStatus::kInfeasible) {
  //   runtime.status = ProblemStatus::INFEASIBLE;
  //   return;
  // }

  // assert(initialactive.size() + initialinactive.size() ==
  //        runtime.instance.num_var);

  // for (HighsInt ia : initialinactive) {
  //     assert(runtime.instance.con_lo[ia] ==
  //            -std::numeric_limits<double>::infinity());
  //     assert(runtime.instance.con_up[ia] ==
  //            std::numeric_limits<double>::infinity());
  //   } else {
  //     // printf("free col %d\n", (int)ia);
  //     assert(runtime.instance.var_lo[ia - runtime.instance.num_con] ==
  //            -std::numeric_limits<double>::infinity());
  //     assert(runtime.instance.var_up[ia - runtime.instance.num_con] ==
  //            std::numeric_limits<double>::infinity());
  //   }
  // }

  // result.rowstatus = atlower;
  // result.active = initialactive;
  // result.inactive = initialinactive;
  // result.primal = x0;
  // result.rowact = ra;
}

#endif
