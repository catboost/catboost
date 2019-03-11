#include "ctr_helper.h"

namespace NCatboostCuda {
    template class TCalcCtrHelper<NCudaLib::TSingleMapping>;
    template class TCalcCtrHelper<NCudaLib::TMirrorMapping>;

    TCtrTargets<NCudaLib::TSingleMapping> DeviceView(const TCtrTargets<NCudaLib::TMirrorMapping>& mirrorTargets, ui32 devId) {
        TCtrTargets<NCudaLib::TSingleMapping> view;
        view.WeightedTarget = mirrorTargets.WeightedTarget.DeviceView(devId);
        view.BinarizedTarget = mirrorTargets.BinarizedTarget.DeviceView(devId);
        view.Weights = mirrorTargets.Weights.DeviceView(devId);
        if (mirrorTargets.HasGroupIds()) {
            view.GroupIds = mirrorTargets.GroupIds.DeviceView(devId);
        }

        view.TotalWeight = mirrorTargets.TotalWeight;
        view.LearnSlice = mirrorTargets.LearnSlice;
        view.TestSlice = mirrorTargets.TestSlice;
        return view;
    }
}
