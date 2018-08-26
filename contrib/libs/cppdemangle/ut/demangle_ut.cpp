#include <contrib/libs/cppdemangle/demangle.h>

#include <library/unittest/registar.h>

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

void Check(TString symbol, TString expectedName) {
    THolder<char, TFree> name = llvm_demangle_gnu3(~symbol);
    TString actualName = name.Get();
    UNIT_ASSERT_VALUES_EQUAL(expectedName, actualName);
}

Y_UNIT_TEST_SUITE(Demangle) {
    Y_UNIT_TEST(Simple1) {
        Check("_ZN2na2nb2nc2caD0Ev", "na::nb::nc::ca::~ca()");
    }

    Y_UNIT_TEST(Simple2) {
        Check("_ZN2na2nb2caI2cbNS_2nc2cbEFS2_RKS4_EEENS_2ccIT_EENS_2cdIT0_EERKNS_2ceIT1_EE", "na::cc<cb> na::nb::ca<cb, na::nc::cb, cb (na::nc::cb const&)>(na::cd<na::nc::cb>, na::ce<cb ()(na::nc::cb const&)> const&)");
    }

    Y_UNIT_TEST(Difficult1) {
        Check("_ZNSt4__y16vectorIN2na2caINS1_2nb2cbEEENS_9allocatorIS5_EEE6assignIPS5_EENS_9enable_ifIXaasr21__is_forward_iteratorIT_EE5valuesr16is_constructibleIS5_NS_15iterator_traitsISC_E9referenceEEE5valueEvE4typeESC_SC_", "std::__y1::enable_if<(__is_forward_iterator<na::ca<na::nb::cb>*>::value) && (is_constructible<na::ca<na::nb::cb>, std::__y1::iterator_traits<na::ca<na::nb::cb>*>::reference>::value), void>::type std::__y1::vector<na::ca<na::nb::cb>, std::__y1::allocator<na::ca<na::nb::cb> > >::assign<na::ca<na::nb::cb>*>(na::ca<na::nb::cb>*, na::ca<na::nb::cb>*)");
    }

    Y_UNIT_TEST(Difficult2) {
        Check("_ZTSN2na2nb2caINS0_2cbIZZNS_2nc2cc2cd2maERKNS_2ceINS_2nd2cfINS_2ne2nf2cgEEEEERKNS6_INS9_2chEEEENKUlPT_E_clINSA_2ciEEEDaSL_EUlvE_EEFvvESR_EE", "typeinfo name for na::nb::ca<na::nb::cb<auto na::nc::cc::cd::ma(na::ce<na::nd::cf<na::ne::nf::cg> > const&, na::ce<na::ne::ch> const&)::'lambda'(auto*)::operator()<na::ne::nf::ci>(auto*) const::'lambda'()>, void (), void ()>");
    }
}
