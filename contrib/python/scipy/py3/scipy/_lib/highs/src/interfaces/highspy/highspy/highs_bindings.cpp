#include "Highs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace pybind11::literals;


void highs_passModel(Highs* h, HighsModel& model)
{
  HighsStatus status = h->passModel(model);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing model");
}
 
void highs_passModelPointers(Highs* h, 
			     const int num_col, const int num_row, const int num_nz,
			     const int q_num_nz, const int a_format, const int q_format,
			     const int sense, const double offset,
			     const py::array_t<double> col_cost,
			     const py::array_t<double> col_lower,
			     const py::array_t<double> col_upper,
			     const py::array_t<double> row_lower,
			     const py::array_t<double> row_upper,
			     const py::array_t<int> a_start,
			     const py::array_t<int> a_index,
			     const py::array_t<double> a_value,
			     const py::array_t<int> q_start,
			     const py::array_t<int> q_index,
			     const py::array_t<double> q_value,
			     const py::array_t<int> integrality)
{
  py::buffer_info col_cost_info = col_cost.request();
  py::buffer_info col_lower_info = col_lower.request();
  py::buffer_info col_upper_info = col_upper.request();
  py::buffer_info row_lower_info = row_lower.request();
  py::buffer_info row_upper_info = row_upper.request();
  py::buffer_info a_start_info = a_start.request();
  py::buffer_info a_index_info = a_index.request();
  py::buffer_info a_value_info = a_value.request();
  py::buffer_info q_start_info = q_start.request();
  py::buffer_info q_index_info = q_index.request();
  py::buffer_info q_value_info = q_value.request();
  py::buffer_info integrality_info = integrality.request();

  const double* col_cost_ptr = static_cast<double*>(col_cost_info.ptr);
  const double* col_lower_ptr = static_cast<double*>(col_lower_info.ptr);
  const double* col_upper_ptr = static_cast<double*>(col_upper_info.ptr);
  const double* row_lower_ptr = static_cast<double*>(row_lower_info.ptr);
  const double* row_upper_ptr = static_cast<double*>(row_upper_info.ptr);
  const int* a_start_ptr = static_cast<int*>(a_start_info.ptr);
  const int* a_index_ptr = static_cast<int*>(a_index_info.ptr);
  const double* a_value_ptr = static_cast<double*>(a_value_info.ptr);
  const int* q_start_ptr = static_cast<int*>(q_start_info.ptr);
  const int* q_index_ptr = static_cast<int*>(q_index_info.ptr);
  const double* q_value_ptr = static_cast<double*>(q_value_info.ptr);
  const int* integrality_ptr = static_cast<int*>(integrality_info.ptr);

  HighsStatus status = h->passModel(num_col, num_row, num_nz,
				    q_num_nz, a_format, q_format,
				    sense, offset, col_cost_ptr,
				    col_lower_ptr, col_upper_ptr, row_lower_ptr,
				    row_upper_ptr, a_start_ptr, a_index_ptr,
				    a_value_ptr, q_start_ptr, q_index_ptr,
				    q_value_ptr, integrality_ptr);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing model");
}
 
void highs_passLp(Highs* h, HighsLp& lp)
{
  HighsStatus status = h->passModel(lp);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing LP");
}
 
void highs_passLpPointers(Highs* h, 
			  const int num_col, const int num_row, const int num_nz,
			  const int a_format, const int sense, const double offset,
			  const py::array_t<double> col_cost,
			  const py::array_t<double> col_lower,
			  const py::array_t<double> col_upper,
			  const py::array_t<double> row_lower,
			  const py::array_t<double> row_upper,
			  const py::array_t<int> a_start,
			  const py::array_t<int> a_index,
			  const py::array_t<double> a_value,
			  const py::array_t<int> integrality)
{
  py::buffer_info col_cost_info = col_cost.request();
  py::buffer_info col_lower_info = col_lower.request();
  py::buffer_info col_upper_info = col_upper.request();
  py::buffer_info row_lower_info = row_lower.request();
  py::buffer_info row_upper_info = row_upper.request();
  py::buffer_info a_start_info = a_start.request();
  py::buffer_info a_index_info = a_index.request();
  py::buffer_info a_value_info = a_value.request();
  py::buffer_info integrality_info = integrality.request();

  const double* col_cost_ptr = static_cast<double*>(col_cost_info.ptr);
  const double* col_lower_ptr = static_cast<double*>(col_lower_info.ptr);
  const double* col_upper_ptr = static_cast<double*>(col_upper_info.ptr);
  const double* row_lower_ptr = static_cast<double*>(row_lower_info.ptr);
  const double* row_upper_ptr = static_cast<double*>(row_upper_info.ptr);
  const int* a_start_ptr = static_cast<int*>(a_start_info.ptr);
  const int* a_index_ptr = static_cast<int*>(a_index_info.ptr);
  const double* a_value_ptr = static_cast<double*>(a_value_info.ptr);
  const int* integrality_ptr = static_cast<int*>(integrality_info.ptr);

  HighsStatus status = h->passModel(num_col, num_row, num_nz,
				    a_format, sense, offset,
				    col_cost_ptr, col_lower_ptr, col_upper_ptr,
				    row_lower_ptr, row_upper_ptr,
				    a_start_ptr, a_index_ptr, a_value_ptr,
				    integrality_ptr);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing model");
}
 
void highs_passHessian(Highs* h, HighsHessian& hessian)
{
  HighsStatus status = h->passHessian(hessian);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing Hessian");
}
 
void highs_passHessianPointers(Highs* h, 
			       const int dim, const int num_nz, const int format,
			       const py::array_t<int> q_start,
			       const py::array_t<int> q_index,
			       const py::array_t<double> q_value)
{
  py::buffer_info q_start_info = q_start.request();
  py::buffer_info q_index_info = q_index.request();
  py::buffer_info q_value_info = q_value.request();

  const int* q_start_ptr = static_cast<int*>(q_start_info.ptr);
  const int* q_index_ptr = static_cast<int*>(q_index_info.ptr);
  const double* q_value_ptr = static_cast<double*>(q_value_info.ptr);

  HighsStatus status = h->passHessian(dim, num_nz, format,
				      q_start_ptr, q_index_ptr, q_value_ptr);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when passing Hessian");
}
 
void highs_writeSolution(Highs* h, const std::string filename, const int style)
{
  HighsStatus status = h->writeSolution(filename, style);
  if (status != HighsStatus::kOk)
    throw py::value_error("Error when writing solution");
}

HighsModelStatus highs_getModelStatus(Highs* h)
{
  return h->getModelStatus(); 
}

bool highs_getDualRay(Highs* h, py::array_t<double> values)
{
  bool has_dual_ray;
  py::buffer_info values_info = values.request();
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->getDualRay(has_dual_ray, values_ptr); 

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when calling get dual ray");

  return has_dual_ray;
}

void highs_addRow(Highs* h, double lower, double upper, int num_new_nz, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addRow(lower, upper, num_new_nz, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding row");
}


void highs_addRows(Highs* h, int num_cons, py::array_t<double> lower, py::array_t<double> upper, int num_new_nz,
		   py::array_t<int> starts, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();
  py::buffer_info starts_info = starts.request();
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);
  int* starts_ptr = static_cast<int*>(starts_info.ptr);
  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addRows(num_cons, lower_ptr, upper_ptr, num_new_nz, starts_ptr, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding rows");
}


void highs_addCol(Highs* h, double cost, double lower, double upper, int num_new_nz, py::array_t<int> indices, py::array_t<double> values)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info values_info = values.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* values_ptr = static_cast<double*>(values_info.ptr);

  HighsStatus status = h->addCol(cost, lower, upper, num_new_nz, indices_ptr, values_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding col");
}


void highs_addVar(Highs* h, double lower, double upper)
{
  HighsStatus status = h->addVar(lower, upper);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding var");  
}


void highs_addVars(Highs* h, int num_vars, py::array_t<double> lower, py::array_t<double> upper)
{
  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();

  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);

  HighsStatus status = h->addVars(num_vars, lower_ptr, upper_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when adding vars");  
}


void highs_changeColsCost(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<double> cost)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info cost_info = cost.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* cost_ptr = static_cast<double*>(cost_info.ptr);

  HighsStatus status = h->changeColsCost(num_set_entries, indices_ptr, cost_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing objective coefficients");  
}


void highs_changeColsBounds(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<double> lower, py::array_t<double> upper)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info lower_info = lower.request();
  py::buffer_info upper_info = upper.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  double* lower_ptr = static_cast<double*>(lower_info.ptr);
  double* upper_ptr = static_cast<double*>(upper_info.ptr);

  HighsStatus status = h->changeColsBounds(num_set_entries, indices_ptr, lower_ptr, upper_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing variable bounds");  
}


void highs_changeColsIntegrality(Highs* h, int num_set_entries, py::array_t<int> indices, py::array_t<HighsVarType> integrality)
{
  py::buffer_info indices_info = indices.request();
  py::buffer_info integrality_info = integrality.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);
  HighsVarType* integrality_ptr = static_cast<HighsVarType*>(integrality_info.ptr);

  HighsStatus status = h->changeColsIntegrality(num_set_entries, indices_ptr, integrality_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when changing variable integrality");  
}


void highs_deleteVars(Highs* h, int num_set_entries, py::array_t<int> indices)
{
  py::buffer_info indices_info = indices.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);

  HighsStatus status = h->deleteVars(num_set_entries, indices_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when deleting columns");  
}


void highs_deleteRows(Highs* h, int num_set_entries, py::array_t<int> indices)
{
  py::buffer_info indices_info = indices.request();

  int* indices_ptr = static_cast<int*>(indices_info.ptr);

  HighsStatus status = h->deleteRows(num_set_entries, indices_ptr);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error when deleting rows");  
}


bool highs_getBoolOption(Highs* h, const std::string& option)
{
  bool res;
  HighsStatus status = h->getOptionValue(option, res);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting option " + option);

  return res;
}


int highs_getIntOption(Highs* h, const std::string& option)
{
  int res;
  HighsStatus status = h->getOptionValue(option, res);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting option " + option);

  return res;
}


double highs_getDoubleOption(Highs* h, const std::string& option)
{
  double res;
  HighsStatus status = h->getOptionValue(option, res);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting option " + option);

  return res;
}


std::string highs_getStringOption(Highs* h, const std::string& option)
{
  std::string res;
  HighsStatus status = h->getOptionValue(option, res);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting option " + option);

  return res;
}


py::object highs_getOptionValue(Highs* h, const std::string& option)
{
  HighsOptionType option_type;
  HighsStatus status = h->getOptionType(option, option_type);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting option " + option);

  if (option_type == HighsOptionType::kBool)
    return py::cast(highs_getBoolOption(h, option));
  else if (option_type == HighsOptionType::kInt)
    return py::cast(highs_getIntOption(h, option));
  else if (option_type == HighsOptionType::kDouble)
    return py::cast(highs_getDoubleOption(h, option));
  else if (option_type == HighsOptionType::kString)
    return py::cast(highs_getStringOption(h, option));
  else
    throw py::value_error("Unrecognized option type");
}


ObjSense highs_getObjectiveSense(Highs* h)
{
  ObjSense obj_sense;
  HighsStatus status = h->getObjectiveSense(obj_sense);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting objective sense");

  return obj_sense;
}


double highs_getObjectiveOffset(Highs* h)
{
  double obj_offset;
  HighsStatus status = h->getObjectiveOffset(obj_offset);

  if (status != HighsStatus::kOk)
    throw py::value_error("Error while getting objective offset");

  return obj_offset;
}


class CallbackTuple {
public:
  CallbackTuple() = default;
  CallbackTuple(py::object _callback, py::object _cb_data) : callback(_callback), callback_data(_cb_data) {}
  ~CallbackTuple() = default;
  py::object callback;
  py::object callback_data;
};


void py_log_callback(HighsLogType log_type, const char* msgbuffer, void* callback_data)
{
  CallbackTuple* callback_tuple = static_cast<CallbackTuple*>(callback_data);
  std::string msg(msgbuffer);
  callback_tuple->callback(log_type, msg, callback_tuple->callback_data);
}


HighsStatus highs_setLogCallback(Highs* h, CallbackTuple* callback_tuple)
{
  void (*_log_callback)(HighsLogType, const char*, void*) = &py_log_callback;
  HighsStatus status = h->setLogCallback(_log_callback, callback_tuple);
  return status;
}


PYBIND11_MODULE(highs_bindings, m)
{
  py::enum_<ObjSense>(m, "ObjSense")
    .value("kMinimize", ObjSense::kMinimize)
    .value("kMaximize", ObjSense::kMaximize);
  py::enum_<MatrixFormat>(m, "MatrixFormat")
    .value("kColwise", MatrixFormat::kColwise)
    .value("kRowwise", MatrixFormat::kRowwise)
    .value("kRowwisePartitioned", MatrixFormat::kRowwisePartitioned);
  py::enum_<HessianFormat>(m, "HessianFormat")
    .value("kTriangular", HessianFormat::kTriangular)
    .value("kSquare", HessianFormat::kSquare);
  py::enum_<SolutionStatus>(m, "SolutionStatus")
    .value("kSolutionStatusNone", SolutionStatus::kSolutionStatusNone)
    .value("kSolutionStatusInfeasible", SolutionStatus::kSolutionStatusInfeasible)
    .value("kSolutionStatusFeasible", SolutionStatus::kSolutionStatusFeasible);
  py::enum_<BasisValidity>(m, "BasisValidity")
    .value("kBasisValidityInvalid", BasisValidity::kBasisValidityInvalid)
    .value("kBasisValidityValid", BasisValidity::kBasisValidityValid);
  py::enum_<HighsModelStatus>(m, "HighsModelStatus")
    .value("kNotset", HighsModelStatus::kNotset)
    .value("kLoadError", HighsModelStatus::kLoadError)
    .value("kModelError", HighsModelStatus::kModelError)
    .value("kPresolveError", HighsModelStatus::kPresolveError)
    .value("kSolveError", HighsModelStatus::kSolveError)
    .value("kPostsolveError", HighsModelStatus::kPostsolveError)
    .value("kModelEmpty", HighsModelStatus::kModelEmpty)
    .value("kOptimal", HighsModelStatus::kOptimal)
    .value("kInfeasible", HighsModelStatus::kInfeasible)
    .value("kUnboundedOrInfeasible", HighsModelStatus::kUnboundedOrInfeasible)
    .value("kUnbounded", HighsModelStatus::kUnbounded)
    .value("kObjectiveBound", HighsModelStatus::kObjectiveBound)
    .value("kObjectiveTarget", HighsModelStatus::kObjectiveTarget)
    .value("kTimeLimit", HighsModelStatus::kTimeLimit)
    .value("kIterationLimit", HighsModelStatus::kIterationLimit)
    .value("kUnknown", HighsModelStatus::kUnknown);
  py::enum_<HighsBasisStatus>(m, "HighsBasisStatus")
    .value("kLower", HighsBasisStatus::kLower)
    .value("kBasic", HighsBasisStatus::kBasic)
    .value("kUpper", HighsBasisStatus::kUpper)
    .value("kZero", HighsBasisStatus::kZero)
    .value("kNonbasic", HighsBasisStatus::kNonbasic);
  py::enum_<HighsVarType>(m, "HighsVarType")
    .value("kContinuous", HighsVarType::kContinuous)
    .value("kInteger", HighsVarType::kInteger)
    .value("kSemiContinuous", HighsVarType::kSemiContinuous)
    .value("kSemiInteger", HighsVarType::kSemiInteger);
  py::enum_<HighsStatus>(m, "HighsStatus")
    .value("kError", HighsStatus::kError)
    .value("kOk", HighsStatus::kOk)
    .value("kWarning", HighsStatus::kWarning);
  py::enum_<HighsLogType>(m, "HighsLogType")
    .value("kInfo", HighsLogType::kInfo)
    .value("kDetailed", HighsLogType::kDetailed)
    .value("kVerbose", HighsLogType::kVerbose)
    .value("kWarning", HighsLogType::kWarning)
    .value("kError", HighsLogType::kError);
  py::class_<CallbackTuple>(m, "CallbackTuple")
    .def(py::init<>())
    .def(py::init<py::object, py::object>())
    .def_readwrite("callback", &CallbackTuple::callback)
    .def_readwrite("callback_data", &CallbackTuple::callback_data);
  py::class_<HighsSparseMatrix>(m, "HighsSparseMatrix")
    .def(py::init<>())
    .def_readwrite("format_", &HighsSparseMatrix::format_)
    .def_readwrite("num_col_", &HighsSparseMatrix::num_col_)
    .def_readwrite("num_row_", &HighsSparseMatrix::num_row_)
    .def_readwrite("start_", &HighsSparseMatrix::start_)
    .def_readwrite("p_end_", &HighsSparseMatrix::p_end_)
    .def_readwrite("index_", &HighsSparseMatrix::index_)
    .def_readwrite("value_", &HighsSparseMatrix::value_);
  py::class_<HighsLp>(m, "HighsLp")
    .def(py::init<>())
    .def_readwrite("num_col_", &HighsLp::num_col_)
    .def_readwrite("num_row_", &HighsLp::num_row_)
    .def_readwrite("col_cost_", &HighsLp::col_cost_)
    .def_readwrite("col_lower_", &HighsLp::col_lower_)
    .def_readwrite("col_upper_", &HighsLp::col_upper_)
    .def_readwrite("row_lower_", &HighsLp::row_lower_)
    .def_readwrite("row_upper_", &HighsLp::row_upper_)
    .def_readwrite("a_matrix_", &HighsLp::a_matrix_)
    .def_readwrite("sense_", &HighsLp::sense_)
    .def_readwrite("offset_", &HighsLp::offset_)
    .def_readwrite("model_name_", &HighsLp::model_name_)
    .def_readwrite("col_names_", &HighsLp::col_names_)
    .def_readwrite("row_names_", &HighsLp::row_names_)
    .def_readwrite("integrality_", &HighsLp::integrality_)
    .def_readwrite("scale_", &HighsLp::scale_)
    .def_readwrite("is_scaled_", &HighsLp::is_scaled_)
    .def_readwrite("is_moved_", &HighsLp::is_moved_)
    .def_readwrite("mods_", &HighsLp::mods_);
  py::class_<HighsHessian>(m, "HighsHessian")
    .def(py::init<>())
    .def_readwrite("dim_", &HighsHessian::dim_)
    .def_readwrite("format_", &HighsHessian::format_)
    .def_readwrite("start_", &HighsHessian::start_)
    .def_readwrite("index_", &HighsHessian::index_)
    .def_readwrite("value_", &HighsHessian::value_);
  py::class_<HighsModel>(m, "HighsModel")
    .def(py::init<>())
    .def_readwrite("lp_", &HighsModel::lp_)
    .def_readwrite("hessian_", &HighsModel::hessian_);
  py::class_<HighsSolution>(m, "HighsSolution")
    .def(py::init<>())
    .def_readwrite("value_valid", &HighsSolution::value_valid)
    .def_readwrite("dual_valid", &HighsSolution::dual_valid)
    .def_readwrite("col_value", &HighsSolution::col_value)
    .def_readwrite("col_dual", &HighsSolution::col_dual)
    .def_readwrite("row_value", &HighsSolution::row_value)
    .def_readwrite("row_dual", &HighsSolution::row_dual);
  py::class_<HighsBasis>(m, "HighsBasis")
    .def(py::init<>())
    .def_readwrite("valid", &HighsBasis::valid)
    .def_readwrite("alien", &HighsBasis::alien)
    .def_readwrite("was_alien", &HighsBasis::was_alien)
    .def_readwrite("debug_id", &HighsBasis::debug_id)
    .def_readwrite("debug_update_count", &HighsBasis::debug_update_count)
    .def_readwrite("debug_origin_name", &HighsBasis::debug_origin_name)
    .def_readwrite("col_status", &HighsBasis::col_status)
    .def_readwrite("row_status", &HighsBasis::row_status);
  py::class_<HighsInfo>(m, "HighsInfo")
    .def(py::init<>())
    .def_readwrite("valid", &HighsInfo::valid)
    .def_readwrite("mip_node_count", &HighsInfo::mip_node_count)
    .def_readwrite("simplex_iteration_count", &HighsInfo::simplex_iteration_count)
    .def_readwrite("ipm_iteration_count", &HighsInfo::ipm_iteration_count)
    .def_readwrite("qp_iteration_count", &HighsInfo::qp_iteration_count)
    .def_readwrite("crossover_iteration_count", &HighsInfo::crossover_iteration_count)
    .def_readwrite("primal_solution_status", &HighsInfo::primal_solution_status)
    .def_readwrite("dual_solution_status", &HighsInfo::dual_solution_status)
    .def_readwrite("basis_validity", &HighsInfo::basis_validity)
    .def_readwrite("objective_function_value", &HighsInfo::objective_function_value)
    .def_readwrite("mip_dual_bound", &HighsInfo::mip_dual_bound)
    .def_readwrite("mip_gap", &HighsInfo::mip_gap)
    .def_readwrite("max_integrality_violation", &HighsInfo::max_integrality_violation)
    .def_readwrite("num_primal_infeasibilities", &HighsInfo::num_primal_infeasibilities)
    .def_readwrite("max_primal_infeasibility", &HighsInfo::max_primal_infeasibility)
    .def_readwrite("sum_primal_infeasibilities", &HighsInfo::sum_primal_infeasibilities)
    .def_readwrite("num_dual_infeasibilities", &HighsInfo::num_dual_infeasibilities)
    .def_readwrite("max_dual_infeasibility", &HighsInfo::max_dual_infeasibility)
    .def_readwrite("sum_dual_infeasibilities", &HighsInfo::sum_dual_infeasibilities);
  py::class_<HighsOptions>(m, "HighsOptions")
    .def(py::init<>())
    .def_readwrite("presolve", &HighsOptions::presolve)
    .def_readwrite("solver", &HighsOptions::solver)
    .def_readwrite("parallel", &HighsOptions::parallel)
    .def_readwrite("ranging", &HighsOptions::ranging)
    .def_readwrite("time_limit", &HighsOptions::time_limit)
    .def_readwrite("infinite_cost", &HighsOptions::infinite_cost)
    .def_readwrite("infinite_bound", &HighsOptions::infinite_bound)
    .def_readwrite("small_matrix_value", &HighsOptions::small_matrix_value)
    .def_readwrite("large_matrix_value", &HighsOptions::large_matrix_value)
    .def_readwrite("primal_feasibility_tolerance", &HighsOptions::primal_feasibility_tolerance)
    .def_readwrite("dual_feasibility_tolerance", &HighsOptions::dual_feasibility_tolerance)
    .def_readwrite("ipm_optimality_tolerance", &HighsOptions::ipm_optimality_tolerance)
    .def_readwrite("objective_bound", &HighsOptions::objective_bound)
    .def_readwrite("objective_target", &HighsOptions::objective_target)
    .def_readwrite("random_seed", &HighsOptions::random_seed)
    .def_readwrite("threads", &HighsOptions::threads)
    .def_readwrite("highs_debug_level", &HighsOptions::highs_debug_level)
    .def_readwrite("highs_analysis_level", &HighsOptions::highs_analysis_level)
    .def_readwrite("simplex_strategy", &HighsOptions::simplex_strategy)
    .def_readwrite("simplex_scale_strategy", &HighsOptions::simplex_scale_strategy)
    .def_readwrite("simplex_crash_strategy", &HighsOptions::simplex_crash_strategy)
    .def_readwrite("simplex_dual_edge_weight_strategy", &HighsOptions::simplex_dual_edge_weight_strategy)
    .def_readwrite("simplex_primal_edge_weight_strategy", &HighsOptions::simplex_primal_edge_weight_strategy)
    .def_readwrite("simplex_iteration_limit", &HighsOptions::simplex_iteration_limit)
    .def_readwrite("simplex_update_limit", &HighsOptions::simplex_update_limit)
    .def_readwrite("simplex_min_concurrency", &HighsOptions::simplex_min_concurrency)
    .def_readwrite("simplex_max_concurrency", &HighsOptions::simplex_max_concurrency)
    .def_readwrite("ipm_iteration_limit", &HighsOptions::ipm_iteration_limit)
    .def_readwrite("write_model_file", &HighsOptions::write_model_file)
    .def_readwrite("solution_file", &HighsOptions::solution_file)
    .def_readwrite("log_file", &HighsOptions::log_file)
    .def_readwrite("write_model_to_file", &HighsOptions::write_model_to_file)
    .def_readwrite("write_solution_to_file", &HighsOptions::write_solution_to_file)
    .def_readwrite("write_solution_style", &HighsOptions::write_solution_style)
    .def_readwrite("output_flag", &HighsOptions::output_flag)
    .def_readwrite("log_to_console", &HighsOptions::log_to_console)
    .def_readwrite("log_dev_level", &HighsOptions::log_dev_level)
    .def_readwrite("run_crossover", &HighsOptions::run_crossover)
    .def_readwrite("allow_unbounded_or_infeasible", &HighsOptions::allow_unbounded_or_infeasible)
    .def_readwrite("allowed_matrix_scale_factor", &HighsOptions::allowed_matrix_scale_factor)
    .def_readwrite("simplex_dualise_strategy", &HighsOptions::simplex_dualise_strategy)
    .def_readwrite("simplex_permute_strategy", &HighsOptions::simplex_permute_strategy)
    .def_readwrite("simplex_price_strategy", &HighsOptions::simplex_price_strategy)
    .def_readwrite("mip_detect_symmetry", &HighsOptions::mip_detect_symmetry)
    .def_readwrite("mip_max_nodes", &HighsOptions::mip_max_nodes)
    .def_readwrite("mip_max_stall_nodes", &HighsOptions::mip_max_stall_nodes)
    .def_readwrite("mip_max_leaves", &HighsOptions::mip_max_leaves)
    .def_readwrite("mip_max_improving_sols", &HighsOptions::mip_max_improving_sols)
    .def_readwrite("mip_lp_age_limit", &HighsOptions::mip_lp_age_limit)
    .def_readwrite("mip_pool_age_limit", &HighsOptions::mip_pool_age_limit)
    .def_readwrite("mip_pool_soft_limit", &HighsOptions::mip_pool_soft_limit)
    .def_readwrite("mip_pscost_minreliable", &HighsOptions::mip_pscost_minreliable)
    .def_readwrite("mip_min_cliquetable_entries_for_parallelism", &HighsOptions::mip_min_cliquetable_entries_for_parallelism)
    .def_readwrite("mip_report_level", &HighsOptions::mip_report_level)
    .def_readwrite("mip_feasibility_tolerance", &HighsOptions::mip_feasibility_tolerance)
    .def_readwrite("mip_rel_gap", &HighsOptions::mip_rel_gap)
    .def_readwrite("mip_abs_gap", &HighsOptions::mip_abs_gap)
    .def_readwrite("mip_heuristic_effort", &HighsOptions::mip_heuristic_effort);
  py::class_<Highs>(m, "_Highs")
    .def(py::init<>())
    .def("passModel", &highs_passModel)
    .def("passModel", &highs_passModelPointers)
    .def("passModel", &highs_passLp)
    .def("passModel", &highs_passLpPointers)
    .def("passHessian", &highs_passHessian)
    .def("passHessian", &highs_passHessianPointers)
    .def("readModel", &Highs::readModel)
    .def("presolve", &Highs::presolve)
    .def("run", &Highs::run)
    .def("postsolve", &Highs::postsolve)
    .def("writeSolution", &highs_writeSolution)
    .def("readSolution", &Highs::readSolution)
    .def("writeModel", &Highs::writeModel)
    .def("getPresolvedLp", &Highs::getPresolvedLp)
    .def("getPresolvedModel", &Highs::getPresolvedModel)
    .def("getModel", &Highs::getModel)
    .def("getLp", &Highs::getLp)
    .def("getSolution", &Highs::getSolution)
    .def("getBasis", &Highs::getBasis)
    .def("getInfo", &Highs::getInfo)
    .def("getRunTime", &Highs::getRunTime)
    .def("getInfinity", &Highs::getInfinity)
    .def("crossover", &Highs::crossover)
    .def("changeObjectiveSense", &Highs::changeObjectiveSense)
    .def("changeObjectiveOffset", &Highs::changeObjectiveOffset)
    .def("changeColIntegrality", &Highs::changeColIntegrality)
    .def("changeColCost", &Highs::changeColCost)
    .def("changeColBounds", &Highs::changeColBounds)
    .def("changeRowBounds", &Highs::changeRowBounds)
    .def("changeCoeff", &Highs::changeCoeff)
    .def("getObjectiveValue", &Highs::getObjectiveValue)
    .def("getObjectiveSense", &highs_getObjectiveSense)
    .def("getObjectiveOffset", &highs_getObjectiveOffset)
    .def("getRunTime", &Highs::getRunTime)
    .def("getModelStatus", &highs_getModelStatus)
    .def("getDualRay", &highs_getDualRay, py::arg("dual_ray_value") = nullptr)
    //    .def("getPrimalRay", &highs_getPrimalRay)   
    //    .def("getObjectiveValue", &Highs::getObjectiveValue)   
    //    .def("getBasicVariables", &highs_getBasicVariables)   
    .def("addRows", &highs_addRows)
    .def("addRow", &highs_addRow)
    .def("addCol", &highs_addCol)
    .def("addVar", &highs_addVar)
    .def("addVars", &highs_addVars)
    .def("changeColsCost", &highs_changeColsCost)
    .def("changeColsBounds", &highs_changeColsBounds)
    .def("changeColsIntegrality", &highs_changeColsIntegrality)
    .def("setLogCallback", &highs_setLogCallback)
    .def("setLogCallback", &highs_setLogCallback)
    .def("deleteVars", &highs_deleteVars)
    .def("deleteRows", &highs_deleteRows)
    .def("clear", &Highs::clear)
    .def("clearModel", &Highs::clearModel)
    .def("clearSolver", &Highs::clearSolver)
    .def("checkSolutionFeasibility", &Highs::checkSolutionFeasibility)
    .def("getNumCol", &Highs::getNumCol)
    .def("getNumRow", &Highs::getNumRow)
    .def("getNumNz", &Highs::getNumNz)
    .def("getHessianNumNz", &Highs::getHessianNumNz)
    .def("resetOptions", &Highs::resetOptions)
    .def("readOptions", &Highs::readOptions)
    .def("passOptions", &Highs::passOptions)
    .def("writeOptions", &Highs::writeOptions, py::arg("filename"), py::arg("report_only_deviations") = false)
    .def("getOptions", &Highs::getOptions)
    .def("getOptionValue", &highs_getOptionValue)
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const bool)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const int)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const double)>(&Highs::setOptionValue))
    .def("setOptionValue", static_cast<HighsStatus (Highs::*)(const std::string&, const std::string&)>(&Highs::setOptionValue))
    .def("writeInfo", &Highs::writeInfo)
    .def("modelStatusToString", &Highs::modelStatusToString)
    .def("solutionStatusToString", &Highs::solutionStatusToString)
    .def("basisStatusToString", &Highs::basisStatusToString)
    .def("basisValidityToString", &Highs::basisValidityToString);
  
  m.attr("kHighsInf") = kHighsInf;
  m.attr("HIGHS_VERSION_MAJOR") = HIGHS_VERSION_MAJOR;
  m.attr("HIGHS_VERSION_MINOR") = HIGHS_VERSION_MINOR;
  m.attr("HIGHS_VERSION_PATCH") = HIGHS_VERSION_PATCH;
}
