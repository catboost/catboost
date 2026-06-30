#include "output.h"
#include <library/cpp/testing/unittest/registar.h>
#include <complex>
#include <sstream>
#include <util/string/builder.h>
#include <util/generic/vector.h>

Y_UNIT_TEST_SUITE(TestOutput) {
    namespace {
        template <typename T>
        void CheckComplexOutHelperImpl(const std::complex<T>& value) {
            // Check Out << works as std::ostream << works
            std::stringstream os;
            os << value;
            std::string stdStr = os.str();
            TString arcadiaStr = TStringBuilder() << value;
            UNIT_ASSERT_VALUES_EQUAL(stdStr, arcadiaStr);
        }

        void CheckComplexOutHelper(float real, float imag) {
            CheckComplexOutHelperImpl(std::complex<double>(real, imag));
            CheckComplexOutHelperImpl(std::complex<float>(real, imag));
        }
    } // namespace

    Y_UNIT_TEST(TestComplex) {
        TVector<float> values({-1., -0.5, 0., 0.5, 1.});
        for (float real : values) {
            for (float imag : values) {
                CheckComplexOutHelper(real, imag);
            }
        }
    }
} // Y_UNIT_TEST_SUITE(TestOutput)
