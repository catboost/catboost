#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>
#include "env.h"

Y_UNIT_TEST_SUITE(EnvTest) {
    Y_UNIT_TEST(GetSetEnvTest) {
        TString key = "util_GETENV_TestVar";
        TString value = "Some value for env var";
        TString def = "Some default value for env var";
        // first of all, it should be clear
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key), TString());
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key, def), def);
        SetEnv(key, value);
        // set and see what value we get here
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key), value);
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key, def), value);
        // set empty value
        SetEnv(key, TString());
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key), TString());

        // check for long values, see IGNIETFERRO-214
        TString longKey = "util_GETENV_TestVarLong";
        TString longValue{1500, 't'};
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(longKey), TString());
        SetEnv(longKey, longValue);
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(longKey), longValue);
        SetEnv(longKey, TString());
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(longKey), TString());
    }

    Y_UNIT_TEST(TryGetEnv) {
        TString key = "util_TryGetEnv_var";
        UNIT_ASSERT_NO_EXCEPTION(TryGetEnv(key));
        SetEnv(key, "value");
        UNIT_ASSERT(TryGetEnv(key).Defined());
        UNIT_ASSERT_VALUES_EQUAL(*TryGetEnv(key), "value");
        UnsetEnv(key);
        UNIT_ASSERT(TryGetEnv(key).Empty());
    }

    Y_UNIT_TEST(UnsetEnv) {
        TString key = "util_UnsetEnv_var";
        SetEnv(key, "value");
        UnsetEnv(key);
        UNIT_ASSERT_VALUES_EQUAL(GetEnv(key, "default_value"), "default_value");
    }

    Y_UNIT_TEST(UnsetNonexistingEnv) {
        TString key = "util_UnsetNonexistingEnv_var";
        UNIT_ASSERT_NO_EXCEPTION(UnsetEnv(key));
        UNIT_ASSERT_NO_EXCEPTION(UnsetEnv(key));
    }

    Y_UNIT_TEST(SetEnvInvalidName) {
        UNIT_ASSERT_EXCEPTION(SetEnv("", "value"), yexception);
        UNIT_ASSERT_EXCEPTION(SetEnv("A=B", "C=D"), yexception);
    }
}
