from .highs_bindings import (
    ObjSense,
    MatrixFormat,
    HessianFormat,
    SolutionStatus,
    BasisValidity,
    HighsModelStatus,
    HighsBasisStatus,
    HighsVarType,
    HighsStatus,
    HighsLogType,
    CallbackTuple,
    HighsSparseMatrix,
    HighsLp,
    HighsHessian,
    HighsModel,
    HighsSolution,
    HighsBasis,
    HighsInfo,
    HighsOptions,
    _Highs,
    kHighsInf,
    HIGHS_VERSION_MAJOR,
    HIGHS_VERSION_MINOR,
    HIGHS_VERSION_PATCH,
)


class Highs(_Highs):
    def __init__(self):
        super().__init__()
        self._log_callback_tuple = CallbackTuple()

    def setLogCallback(self, func, callback_data):
        self._log_callback_tuple.callback = func
        self._log_callback_tuple.callback_data = callback_data
        super().setLogCallback(self._log_callback_tuple)
