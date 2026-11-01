#include <library/cpp/yt/system/access_spreader.h>

namespace {

////////////////////////////////////////////////////////////////////////////////

constexpr int StripeCount = 64;

bool ConstructorInitialized = false;

int InitializeConstructorStripe()
{
    int stripe = NYT::GetCurrentAccessStripe(StripeCount);
    ConstructorInitialized = true;
    return stripe;
}

const int ConstructorStripe = InitializeConstructorStripe();

////////////////////////////////////////////////////////////////////////////////

} // namespace

extern "C" {

////////////////////////////////////////////////////////////////////////////////

int GetCurrentAccessStripeFromNativeLibrary()
{
    return NYT::GetCurrentAccessStripe(StripeCount);
}

int GetConstructorStripe()
{
    return ConstructorStripe;
}

bool GetConstructorInitialized()
{
    return ConstructorInitialized;
}

////////////////////////////////////////////////////////////////////////////////

} // extern "C"
