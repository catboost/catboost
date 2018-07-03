#include "defaults_helper.h"

double Round(double number, int precision) {
    const double multiplier = pow(10, precision);
    return round(number * multiplier) / multiplier;
}
