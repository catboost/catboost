using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

// mcs -out:highscslib.dll -t:library highs_csharp_api.cs -unsafe

public enum HighsStatus
{
   kError = -1,
   kOk,
   kWarning
}

public enum HighsMatrixFormat 
{ 
   kColwise = 1, 
   kRowwise 
}

public enum HighsBasisStatus
{
  kLower = 0,
  kBasic,
  kUpper,
  kZero,
  kNonbasic
}

public enum HighsObjectiveSense
{
   kMinimize = 1,
   kMaximize = -1
}

public enum HighsModelStatus
{
  kNotset = 0,
  kLoadError,
  kModelError,
  kPresolveError,
  kSolveError,
  kPostsolveError,
  kModelEmpty,
  kOptimal,
  kInfeasible,
  kUnboundedOrInfeasible,
  kUnbounded,
  kObjectiveBound,
  kObjectiveTarget,
  kTimeLimit,
  kIterationLimit,
  kUnknown
}

public class HighsModel
{
   public HighsObjectiveSense sense;
   public double[] colcost;
   public double offset;
   public double[] collower;
   public double[] colupper;
   public double[] rowlower;
   public double[] rowupper;
   public HighsMatrixFormat a_format;
   public int[] astart;
   public int[] aindex;
   public double[] avalue;

   public HighsModel()
   {

   }

   public HighsModel(double[] colcost, double[] collower, double[] colupper, double[] rowlower, double[] rowupper,
   int[] astart, int[] aindex, double[] avalue, double offset = 0, HighsMatrixFormat a_format = HighsMatrixFormat.kColwise, HighsObjectiveSense sense = HighsObjectiveSense.kMinimize)
   {
      this.colcost = colcost;
      this.collower = collower;
      this.colupper = colupper;
      this.rowlower = rowlower;
      this.rowupper = rowupper;
      this.astart = astart;
      this.aindex = aindex;
      this.avalue = avalue;
      this.offset = offset;
      this.a_format = a_format;
      this.sense = sense;
   }
}

public class HighsSolution
{
   public double[] colvalue;
   public double[] coldual;
   public double[] rowvalue;
   public double[] rowdual;

   public HighsSolution(int numcol, int numrow)
   {
      this.colvalue = new double[numcol];
      this.coldual = new double[numcol];
      this.rowvalue = new double[numrow];
      this.rowdual = new double[numrow];
   }

   public HighsSolution(double[] colvalue, double[] coldual, double[] rowvalue, double[] rowdual)
   {
      this.colvalue = colvalue;
      this.coldual = coldual;
      this.rowvalue = rowvalue;
      this.rowdual = rowdual;
   }
}

public class HighsBasis
{
   public HighsBasisStatus[] colbasisstatus;
   public HighsBasisStatus[] rowbasisstatus;

   public HighsBasis(int numcol, int numrow)
   {
      this.colbasisstatus = new HighsBasisStatus[numcol];
      this.rowbasisstatus = new HighsBasisStatus[numrow];
   }

   public HighsBasis(HighsBasisStatus[] colbasisstatus, HighsBasisStatus[] rowbasisstatus)
   {
      this.colbasisstatus = colbasisstatus;
      this.rowbasisstatus = rowbasisstatus;
   }
}

public class HighsLpSolver : IDisposable
{
   private IntPtr highs;
   private bool _disposed;

   private const string highslibname = "highs";

   [DllImport(highslibname)]
   private static extern int Highs_call(Int32 numcol, Int32 numrow, Int32 numnz, double[] colcost,
   double[] collower, double[] colupper, double[] rowlower, double[] rowupper, int[] astart, int[] aindex, double[] avalue,
   double[] colvalue, double[] coldual, double[] rowvalue, double[] rowdual, int[] colbasisstatus, int[] rowbasisstatus, ref int modelstatus);

   [DllImport(highslibname)]
   private static extern IntPtr Highs_create();

   [DllImport(highslibname)]
   private static extern void Highs_destroy(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_run(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_readModel(IntPtr highs, string filename);

   [DllImport(highslibname)]
   private static extern int Highs_writeModel(IntPtr highs, string filename);

   [DllImport(highslibname)]
   private static extern int Highs_passLp(IntPtr highs, int numcol, int numrow, int numnz, int aformat, int sense, double offset, double[] colcost,
   double[] collower, double[] colupper, double[] rowlower, double[] rowupper, int[] astart, int[] aindex, double[] avalue);

   [DllImport(highslibname)]
   private static extern int Highs_setOptionValue(IntPtr highs, string option, string value);

   [DllImport(highslibname)]
   private static extern int Highs_setBoolOptionValue(IntPtr highs, string option, int value);

   [DllImport(highslibname)]
   private static extern int Highs_setIntOptionValue(IntPtr highs, string option, int value);

   [DllImport(highslibname)]
   private static extern int Highs_setDoubleOptionValue(IntPtr highs, string option, double value);

   [DllImport(highslibname)]
   private static extern int Highs_setStringOptionValue(IntPtr highs, string option, string value);

   [DllImport(highslibname)]
   private static extern int Highs_getBoolOptionValue(IntPtr highs, string option, out int value);

   [DllImport(highslibname)]
   private static extern int Highs_getIntOptionValue(IntPtr highs, string option, out int value);

   [DllImport(highslibname)]
   private static extern int Highs_getDoubleOptionValue(IntPtr highs, string option, out double value);

   [DllImport(highslibname)]
   private static extern int Highs_getStringOptionValue(IntPtr highs, string option, [Out] StringBuilder value);

   [DllImport(highslibname)]
   private static extern int Highs_getSolution(IntPtr highs, double[] colvalue, double[] coldual, double[] rowvalue, double[] rowdual);

   [DllImport(highslibname)]
   private static extern int Highs_getNumCol(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_getNumRow(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_getNumNz(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_getBasis(IntPtr highs, int[] colstatus, int[] rowstatus);

   [DllImport(highslibname)]
   private static extern double Highs_getObjectiveValue(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_getIterationCount(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_getModelStatus(IntPtr highs);

   [DllImport(highslibname)]
   private static extern int Highs_addRow(IntPtr highs, double lower, double upper, int num_new_nz, int[] indices, double[] values);

   [DllImport(highslibname)]
   private static extern int Highs_addRows(IntPtr highs, int num_new_row, double[] lower, double[] upper,
   int num_new_nz, int[] starts, int[] indices, double[] values);

   [DllImport(highslibname)]
   private static extern int Highs_addCol(IntPtr highs, double cost, double lower, double upper,
   int num_new_nz, int[] indices, double[] values);

   [DllImport(highslibname)]
   private static extern int Highs_addCols(IntPtr highs, int num_new_col, double[] costs, double[] lower, double[] upper,
   int num_new_nz, int[] starts, int[] indices, double[] values);

   [DllImport(highslibname)]
   private static extern int Highs_changeObjectiveSense(IntPtr highs, int sense);

   [DllImport(highslibname)]
   private static extern int Highs_changeColCost(IntPtr highs, int col, double cost);

   [DllImport(highslibname)]
   private static extern int Highs_changeColsCostBySet(IntPtr highs, int num_set_entries, int[] set, double[] cost);

   [DllImport(highslibname)]
   private static extern int Highs_changeColsCostByMask(IntPtr highs, int[] mask, double[] cost);

   [DllImport(highslibname)]
   private static extern int Highs_changeColBounds(IntPtr highs, int col, double lower, double upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeColsBoundsByRange(IntPtr highs, int from_col, int to_col, double[] lower, double[] upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeColsBoundsBySet(IntPtr highs, int num_set_entries, int[] set, double[] lower, double[] upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeColsBoundsByMask(IntPtr highs, int[] mask, double[] lower, double[] upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeRowBounds(IntPtr highs, int row, double lower, double upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeRowsBoundsBySet(IntPtr highs, int num_set_entries, int[] set, double[] lower, double[] upper);

   [DllImport(highslibname)]
   private static extern int Highs_changeRowsBoundsByMask(IntPtr highs, int[] mask, double[] lower, double[] upper);

   [DllImport(highslibname)]
   private static extern int Highs_deleteColsByRange(IntPtr highs, int from_col, int to_col);

   [DllImport(highslibname)]
   private static extern int Highs_deleteColsBySet(IntPtr highs, int num_set_entries, int[] set);

   [DllImport(highslibname)]
   private static extern int Highs_deleteColsByMask(IntPtr highs, int[] mask);

   [DllImport(highslibname)]
   private static extern int Highs_deleteRowsByRange(IntPtr highs, int from_row, int to_row);

   [DllImport(highslibname)]
   private static extern int Highs_deleteRowsBySet(IntPtr highs, int num_set_entries, int[] set);

   [DllImport(highslibname)]
   private static extern int Highs_deleteRowsByMask(IntPtr highs, int[] mask);

   [DllImport(highslibname)]
   private static extern int Highs_getColsByRange(IntPtr highs, int from_col, int to_col, ref int num_col, double[] costs,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getColsBySet(IntPtr highs, int num_set_entries, int[] set, ref int num_col, double[] costs,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getColsByMask(IntPtr highs, int[] mask, ref int num_col, double[] costs,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getRowsByRange(IntPtr highs, int from_row, int to_row, ref int num_row,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getRowsBySet(IntPtr highs, int num_set_entries, int[] set, ref int num_row,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getRowsByMask(IntPtr highs, int[] mask, ref int num_row,
   double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   [DllImport(highslibname)]
   private static extern int Highs_getBasicVariables(IntPtr highs, int[] basic_variables);

   [DllImport(highslibname)]
   private static extern int Highs_getBasisInverseRow(IntPtr highs, int row, double[] row_vector, ref int row_num_nz, int[] row_indices);

   [DllImport(highslibname)]
   private static extern int Highs_getBasisInverseCol(IntPtr highs, int col, double[] col_vector, ref int col_num_nz, int[] col_indices);

   [DllImport(highslibname)]
   private static extern int Highs_getBasisSolve(IntPtr highs, double[] rhs, double[] solution_vector, ref int solution_num_nz, int[] solution_indices);

   [DllImport(highslibname)]
   private static extern int Highs_getBasisTransposeSolve(IntPtr highs, double[] rhs, double[] solution_vector, ref int solution_nz, int[] solution_indices);

   [DllImport(highslibname)]
   private static extern int Highs_getReducedRow(IntPtr highs, int row, double[] row_vector, ref int row_num_nz, int[] row_indices);

   [DllImport(highslibname)]
   private static extern int Highs_getReducedColumn(IntPtr highs, int col, double[] col_vector, ref int col_num_nz, int[] col_indices);

   public static HighsStatus call(HighsModel model, ref HighsSolution sol, ref HighsBasis bas, ref HighsModelStatus modelstatus)
   {
      int nc = model.colcost.Length;
      int nr = model.rowlower.Length;
      int nnz = model.avalue.Length;

      int[] colbasstat = new int[nc];
      int[] rowbasstat = new int[nr];

      int modelstate = 0;

      HighsStatus status = (HighsStatus)HighsLpSolver.Highs_call(nc, nr, nnz, model.colcost, model.collower, model.colupper,
      model.rowlower, model.rowupper, model.astart, model.aindex, model.avalue,
      sol.colvalue, sol.coldual, sol.rowvalue, sol.rowdual, colbasstat, rowbasstat, ref modelstate);

      modelstatus = (HighsModelStatus)modelstate;

      bas.colbasisstatus = colbasstat.Select(x => (HighsBasisStatus)x).ToArray();
      bas.rowbasisstatus = rowbasstat.Select(x => (HighsBasisStatus)x).ToArray();

      return status;
   }

   public HighsLpSolver()
   {
      this.highs = HighsLpSolver.Highs_create();
   }

   ~HighsLpSolver()
   {
      this.Dispose(false);
   }

   public void Dispose()
   {
      this.Dispose(true);
      GC.SuppressFinalize(this);
   }

   protected virtual void Dispose(bool disposing)
   {
      if (this._disposed)
      {
         return;
      }

      HighsLpSolver.Highs_destroy(this.highs);
      this._disposed = true;
   }

   public HighsStatus run()
   {
      return (HighsStatus)HighsLpSolver.Highs_run(this.highs);
   }

   public HighsStatus readModel(string filename)
   {
      return (HighsStatus)HighsLpSolver.Highs_readModel(this.highs, filename);
   }

   public HighsStatus writeModel(string filename)
   {
      return (HighsStatus)HighsLpSolver.Highs_writeModel(this.highs, filename);
   }

   public HighsStatus passLp(HighsModel model)
   {
      return (HighsStatus)HighsLpSolver.Highs_passLp(this.highs, model.colcost.Length, model.rowlower.Length, model.avalue.Length,
      (int)model.a_format, (int)model.sense, model.offset, model.colcost, model.collower, model.colupper, model.rowlower, model.rowupper, model.astart, model.aindex, model.avalue);
   }

   public HighsStatus setOptionValue(string option, string value)
   {
      return (HighsStatus)HighsLpSolver.Highs_setOptionValue(this.highs, option, value);
   }

   public HighsStatus setStringOptionValue(string option, string value)
   {
      return (HighsStatus)HighsLpSolver.Highs_setStringOptionValue(this.highs, option, value);
   }

   public HighsStatus setBoolOptionValue(string option, int value)
   {
      return (HighsStatus)HighsLpSolver.Highs_setBoolOptionValue(this.highs, option, value);
   }

   public HighsStatus setDoubleOptionValue(string option, double value)
   {
      return (HighsStatus)HighsLpSolver.Highs_setDoubleOptionValue(this.highs, option, value);
   }

   public HighsStatus setIntOptionValue(string option, int value)
   {
      return (HighsStatus)HighsLpSolver.Highs_setIntOptionValue(this.highs, option, value);
   }

   public HighsStatus getStringOptionValue(string option, out string value)
   {
      var stringBuilder = new StringBuilder();
      var result = (HighsStatus)HighsLpSolver.Highs_getStringOptionValue(this.highs, option, stringBuilder);
      value = stringBuilder.ToString();
      return result;
   }

   public HighsStatus getBoolOptionValue(string option, out int value)
   {
      return (HighsStatus)HighsLpSolver.Highs_getBoolOptionValue(this.highs, option, out value);
   }

   public HighsStatus getDoubleOptionValue(string option, out double value)
   {
      return (HighsStatus)HighsLpSolver.Highs_getDoubleOptionValue(this.highs, option, out value);
   }

   public HighsStatus getIntOptionValue(string option, out int value)
   {
      return (HighsStatus)HighsLpSolver.Highs_getIntOptionValue(this.highs, option, out value);
   }

   public int getNumCol()
   {
      return HighsLpSolver.Highs_getNumCol(this.highs);
   }

   public int getNumRow()
   {
      return HighsLpSolver.Highs_getNumRow(this.highs);
   }

   public int getNumNz()
   {
      return HighsLpSolver.Highs_getNumNz(this.highs);
   }

   public HighsSolution getSolution()
   {
      int nc = this.getNumCol();
      int nr = this.getNumRow();

      HighsSolution sol = new HighsSolution(nc, nr);
      HighsLpSolver.Highs_getSolution(this.highs, sol.colvalue, sol.coldual, sol.rowvalue, sol.rowdual);
      return sol;
   }

   public HighsBasis getBasis()
   {
      int nc = this.getNumCol();
      int nr = this.getNumRow();

      int[] colbasstat = new int[nc];
      int[] rowbasstat = new int[nr];

      HighsLpSolver.Highs_getBasis(this.highs, colbasstat, rowbasstat);
      HighsBasis bas = new HighsBasis(colbasstat.Select(x => (HighsBasisStatus)x).ToArray(), rowbasstat.Select(x => (HighsBasisStatus)x).ToArray());

      return bas;
   }

   public double getObjectiveValue()
   {
      return HighsLpSolver.Highs_getObjectiveValue(this.highs);
   }

   public HighsModelStatus GetModelStatus()
   {
      return (HighsModelStatus)HighsLpSolver.Highs_getModelStatus(this.highs);
   }

   public int getIterationCount()
   {
      return HighsLpSolver.Highs_getIterationCount(this.highs);
   }

   public HighsStatus addRow(double lower, double upper, int[] indices, double[] values)
   {
      return (HighsStatus)HighsLpSolver.Highs_addRow(this.highs, lower, upper, indices.Length, indices, values);
   }

   public HighsStatus addRows(double[] lower, double[] upper, int[] starts, int[] indices, double[] values)
   {
      return (HighsStatus)HighsLpSolver.Highs_addRows(this.highs, lower.Length, lower, upper, indices.Length, starts, indices, values);
   }

   public HighsStatus addCol(double cost, double lower, double upper, int[] indices, double[] values)
   {
      return (HighsStatus)HighsLpSolver.Highs_addCol(this.highs, cost, lower, upper, indices.Length, indices, values);
   }

   public HighsStatus addCols(double[] costs, double[] lower, double[] upper, int[] starts, int[] indices, double[] values)
   {
      return (HighsStatus)HighsLpSolver.Highs_addCols(this.highs, costs.Length, costs, lower, upper, indices.Length, starts, indices, values);
   }

   public HighsStatus changeObjectiveSense(HighsObjectiveSense sense)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeObjectiveSense(this.highs, (int)sense);
   }

   public HighsStatus changeColCost(int col, double cost)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColCost(this.highs, col, cost);
   }

   public HighsStatus changeColsCostBySet(int[] cols, double[] costs)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColsCostBySet(this.highs, cols.Length, cols, costs);
   }

   public HighsStatus changeColsCostByMask(bool[] mask, double[] cost)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColsCostByMask(this.highs, mask.Select(x => x ? 1 : 0).ToArray(), cost);
   }

   public HighsStatus changeColBounds(int col, double lower, double upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColBounds(this.highs, col, lower, upper);
   }

   public HighsStatus changeColsBoundsByRange(int from, int to, double[] lower, double[] upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColsBoundsByRange(this.highs, from, to, lower, upper);
   }

   public HighsStatus changeColsBoundsBySet(int[] cols, double[] lower, double[] upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColsBoundsBySet(this.highs, cols.Length, cols, lower, upper);
   }

   public HighsStatus changeColsBoundsByMask(bool[] mask, double[] lower, double[] upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeColsBoundsByMask(this.highs, mask.Select(x => x ? 1 : 0).ToArray(), lower, upper);
   }

   public HighsStatus changeRowBounds(int row, double lower, double upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeRowBounds(this.highs, row, lower, upper);
   }

   public HighsStatus changeRowsBoundsBySet(int[] rows, double[] lower, double[] upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeRowsBoundsBySet(this.highs, rows.Length, rows, lower, upper);
   }

   public HighsStatus changeRowsBoundsByMask(bool[] mask, double[] lower, double[] upper)
   {
      return (HighsStatus)HighsLpSolver.Highs_changeRowsBoundsByMask(this.highs, mask.Select(x => x ? 1 : 0).ToArray(), lower, upper);
   }

   public HighsStatus deleteColsByRange(int from, int to)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteColsByRange(this.highs, from, to);
   }

   public HighsStatus deleteColsBySet(int[] cols)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteColsBySet(this.highs, cols.Length, cols);
   }

   public HighsStatus deleteColsByMask(bool[] mask)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteColsByMask(this.highs, mask.Select(x => x ? 1 : 0).ToArray());
   }

   public HighsStatus deleteRowsByRange(int from, int to)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteRowsByRange(this.highs, from, to);
   }

   public HighsStatus deleteRowsBySet(int[] rows)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteRowsBySet(this.highs, rows.Length, rows);
   }

   public HighsStatus deleteRowsByMask(bool[] mask)
   {
      return (HighsStatus)HighsLpSolver.Highs_deleteRowsByMask(this.highs, mask.Select(x => x ? 1 : 0).ToArray());
   }

   // int Highs_getColsByRange(void *highs, int from_col, int to_col, ref int num_col, double[] costs, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   // [DllImport(highslibname)]
   // int Highs_getColsBySet(void *highs, int num_set_entries, int[] set, ref int num_col, double[] costs, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   // [DllImport(highslibname)]
   // int Highs_getColsByMask(void *highs, int[] mask, ref int num_col, double[] costs, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   // [DllImport(highslibname)]
   // int Highs_getRowsByRange(void *highs, int from_row, int to_row, ref int num_row, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   // [DllImport(highslibname)]
   // int Highs_getRowsBySet(void *highs, int num_set_entries, int[] set, ref int num_row, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   // [DllImport(highslibname)]
   // int Highs_getRowsByMask(void *highs, int[] mask, ref int num_row, 
   // double[] lower, double[] upper, ref int num_nz, int[] matrix_start, int[] matrix_index, double[] matrix_value);

   public HighsStatus getBasicVariables(ref int[] basic_variables)
   {
      return (HighsStatus)Highs_getBasicVariables(this.highs, basic_variables);
   }

   public HighsStatus getBasisInverseRow(int row, double[] row_vector, ref int row_num_nz, int[] row_indices)
   {
      return (HighsStatus)Highs_getBasisInverseRow(this.highs, row, row_vector, ref row_num_nz, row_indices);
   }

   public HighsStatus getBasisInverseCol(int col, double[] col_vector, ref int col_num_nz, int[] col_indices)
   {
      return (HighsStatus)Highs_getBasisInverseCol(this.highs, col, col_vector, ref col_num_nz, col_indices);
   }

   public HighsStatus getBasisSolve(double[] rhs, double[] solution_vector, ref int solution_num_nz, int[] solution_indices)
   {
      return (HighsStatus)Highs_getBasisSolve(this.highs, rhs, solution_vector, ref solution_num_nz, solution_indices);
   }

   public HighsStatus getBasisTransposeSolve(double[] rhs, double[] solution_vector, ref int solution_num_nz, int[] solution_indices)
   {
      return (HighsStatus)Highs_getBasisTransposeSolve(this.highs, rhs, solution_vector, ref solution_num_nz, solution_indices);
   }

   public HighsStatus getReducedRow(int row, double[] row_vector, ref int row_num_nz, int[] row_indices)
   {
      return (HighsStatus)Highs_getReducedRow(this.highs, row, row_vector, ref row_num_nz, row_indices);
   }

   public HighsStatus getReducedColumn(int col, double[] col_vector, ref int col_num_nz, int[] col_indices)
   {
      return (HighsStatus)Highs_getReducedColumn(this.highs, col, col_vector, ref col_num_nz, col_indices);
   }
}
