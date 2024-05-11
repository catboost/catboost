import unittest
import highspy
import numpy as np
from pyomo.common.tee import capture_output
from io import StringIO


class TestHighsPy(unittest.TestCase):
    def get_basic_model(self):
        """
        min y
        s.t.
        -x + y >= 2
        x + y >= 0
        """
        inf = highspy.kHighsInf
        h = highspy.Highs()
        h.addVars(2, np.array([-inf, -inf]), np.array([inf, inf]))
        h.changeColsCost(2, np.array([0, 1]), np.array([0, 1], dtype=np.double))
        num_cons = 2
        lower = np.array([2, 0], dtype=np.double)
        upper = np.array([inf, inf], dtype=np.double)
        num_new_nz = 4
        starts = np.array([0, 2])
        indices = np.array([0, 1, 0, 1])
        values = np.array([-1, 1, 1, 1], dtype=np.double)
        h.addRows(num_cons, lower, upper, num_new_nz, starts, indices, values)
        h.setOptionValue('log_to_console', False)
        return h
    
    def get_infeasible_model(self):
        inf = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_ = 2;
        lp.num_row_ = 2;
        lp.col_cost_ = np.array([10, 15], dtype=np.double)
        lp.col_lower_ = np.array([0, 0], dtype=np.double)
        lp.col_upper_ = np.array([inf, inf], dtype=np.double)
        lp.row_lower_ = np.array([3, 1], dtype=np.double)
        lp.row_upper_ = np.array([3, 1], dtype=np.double)
        lp.a_matrix_.start_ = np.array([0, 2, 4])
        lp.a_matrix_.index_ = np.array([0, 1, 0, 1])
        lp.a_matrix_.value_ = np.array([2, 1, 1, 3], dtype=np.double)
        lp.offset_ = 0;
        h = highspy.Highs()
        h.passModel(lp)
        h.setOptionValue('log_to_console', False)
        h.setOptionValue('presolve', 'off')
        return h
    
    def test_basics(self):
        inf = highspy.kHighsInf
        h = self.get_basic_model()
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -1)
        self.assertAlmostEqual(sol.col_value[1], 1)

        """
        min y
        s.t.
        -x + y >= 3
        x + y >= 0
        """
        h.changeRowBounds(0, 3, inf)
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -1.5)
        self.assertAlmostEqual(sol.col_value[1], 1.5)

        # now make y integer
        h.changeColsIntegrality(1, np.array([1]), np.array([highspy.HighsVarType.kInteger]))
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -1)
        self.assertAlmostEqual(sol.col_value[1], 2)

        """
        now delete the first constraint and add a new one
        
        min y
        s.t.
        x + y >= 0
        -x + y >= 0
        """
        h.deleteRows(1, np.array([0]))
        h.addRows(1, np.array([0], dtype=np.double), np.array([inf]), 2,
                  np.array([0]), np.array([0, 1]), np.array([-1, 1], dtype=np.double))
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], 0)
        self.assertAlmostEqual(sol.col_value[1], 0)

        # change the upper bound of x to -5
        h.changeColsBounds(1, np.array([0]), np.array([-inf], dtype=np.double),
                           np.array([-5], dtype=np.double))
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -5)
        self.assertAlmostEqual(sol.col_value[1], 5)

        # now maximize
        h.changeColCost(1, -1)
        h.changeRowBounds(0, -inf, 0)
        h.changeRowBounds(1, -inf, 0)
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -5)
        self.assertAlmostEqual(sol.col_value[1], -5)

        h.changeColCost(1, 1)
        self.assertEqual(h.getObjectiveSense(), highspy.ObjSense.kMinimize)
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        self.assertEqual(h.getObjectiveSense(), highspy.ObjSense.kMaximize)
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -5)
        self.assertAlmostEqual(sol.col_value[1], -5)

        self.assertAlmostEqual(h.getObjectiveValue(), -5)
        h.changeObjectiveOffset(1)
        self.assertAlmostEqual(h.getObjectiveOffset(), 1)
        h.run()
        self.assertAlmostEqual(h.getObjectiveValue(), -4)

    def test_options(self):
        # test bool option
        h = highspy.Highs()
        h.setOptionValue('log_to_console', True)
        self.assertTrue(h.getOptionValue('log_to_console'))
        h.setOptionValue('log_to_console', False)
        self.assertFalse(h.getOptionValue('log_to_console'))

        # test string option
        h.setOptionValue('presolve', 'off')
        self.assertEqual(h.getOptionValue('presolve'), 'off')
        h.setOptionValue('presolve', 'on')
        self.assertEqual(h.getOptionValue('presolve'), 'on')

        # test int option
        h.setOptionValue('threads', 1)
        self.assertEqual(h.getOptionValue('threads'), 1)
        h.setOptionValue('threads', 2)
        self.assertEqual(h.getOptionValue('threads'), 2)

        # test double option
        h.setOptionValue('time_limit', 1.7)
        self.assertAlmostEqual(h.getOptionValue('time_limit'), 1.7)
        h.setOptionValue('time_limit', 2.7)
        self.assertAlmostEqual(h.getOptionValue('time_limit'), 2.7)

    def test_clear(self):
        h = self.get_basic_model()
        self.assertEqual(h.getNumCol(), 2)
        self.assertEqual(h.getNumRow(), 2)
        self.assertEqual(h.getNumNz(), 4)

        orig_feas_tol = h.getOptionValue('primal_feasibility_tolerance')
        new_feas_tol = orig_feas_tol + 1
        h.setOptionValue('primal_feasibility_tolerance', new_feas_tol)
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), new_feas_tol)
        h.clear()
        self.assertEqual(h.getNumCol(), 0)
        self.assertEqual(h.getNumRow(), 0)
        self.assertEqual(h.getNumNz(), 0)
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), orig_feas_tol)

        h = self.get_basic_model()
        h.setOptionValue('primal_feasibility_tolerance', new_feas_tol)
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), new_feas_tol)
        h.clearModel()
        self.assertEqual(h.getNumCol(), 0)
        self.assertEqual(h.getNumRow(), 0)
        self.assertEqual(h.getNumNz(), 0)
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), new_feas_tol)

        h = self.get_basic_model()
        h.run()
        sol = h.getSolution()
        self.assertAlmostEqual(sol.col_value[0], -1)
        self.assertAlmostEqual(sol.col_value[1], 1)
        h.clearSolver()
        self.assertEqual(h.getNumCol(), 2)
        self.assertEqual(h.getNumRow(), 2)
        self.assertEqual(h.getNumNz(), 4)
        sol = h.getSolution()
        self.assertFalse(sol.value_valid)
        self.assertFalse(sol.dual_valid)

        h = self.get_basic_model()
        orig_feas_tol = h.getOptionValue('primal_feasibility_tolerance')
        new_feas_tol = orig_feas_tol + 1
        h.setOptionValue('primal_feasibility_tolerance', new_feas_tol)
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), new_feas_tol)
        h.resetOptions()
        self.assertAlmostEqual(h.getOptionValue('primal_feasibility_tolerance'), orig_feas_tol)

    # def test_dual_ray(self):
    #     h = self.get_infeasible_model()
    #     h.setOptionValue('log_to_console', True)
    #     h.run()
    #     has_dual_ray = h.getDualRay()
    #     print('has_dual_ray = ', has_dual_ray)
    #     self.assertTrue(has_dual_ray)
 
    def test_check_solution_feasibility(self):
        h = self.get_basic_model()
        h.setOptionValue('log_to_console', True)
        h.checkSolutionFeasibility()
        h.run()
        h.checkSolutionFeasibility()

    def test_log_callback(self):
        h = self.get_basic_model()
        h.setOptionValue('log_to_console', True)

        class Foo(object):
            def __str__(self):
                return 'an instance of Foo'

            def __repr__(self):
                return self.__str__()

        def log_callback(log_type, message, data):
            print('got a log message: ', log_type, data, message)

        h.setLogCallback(log_callback, Foo())
        out = StringIO()
        with capture_output(out) as t:
            h.run()
        out = out.getvalue()
        self.assertIn('got a log message:  HighsLogType.kInfo an instance of Foo Presolving model', out)

