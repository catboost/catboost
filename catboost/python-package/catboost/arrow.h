#pragma once

#include <Python.h>

#include <util/generic/fwd.h>
#include <util/system/types.h>

#include <future>


namespace NCB {

    class IRawFeaturesOrderDataVisitor;


    void AsyncAddArrowNumColumn(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    );

    void AsyncAddArrowCategoricalColumnOfStrings(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    );

    void AsyncAddArrowCategoricalColumnOfIntOrBoolean(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    );

    void AsyncAddArrowTextColumn(
        ui32 flatFeatureIdx,
        PyObject* capsule,
        IRawFeaturesOrderDataVisitor* builderVisitor,
        TVector<std::future<void>>* result
    );

}
