#pragma once

template <typename TFunc>
static auto DispatchGenericLambda(TFunc function) {
    return function();
}

template <typename TFunc, typename ...TBools>
static auto DispatchGenericLambda(TFunc function, bool arg, TBools ...args) {
    return DispatchGenericLambda([=](auto ...args) {
        if (arg) {
            return function(std::true_type(), args...);
        } else {
            return function(std::false_type(), args...);
        }
    }, args...);
}
