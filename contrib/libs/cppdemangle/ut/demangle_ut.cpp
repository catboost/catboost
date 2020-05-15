#include <library/cpp/unittest/registar.h>

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/demangle.h>

void Check(TString symbol, TString expectedName) {
    TString actualName = CppDemangle(symbol);
    UNIT_ASSERT_VALUES_EQUAL(expectedName, actualName);
}

Y_UNIT_TEST_SUITE(Demangle) {
    Y_UNIT_TEST(Simple1) {
        Check("_ZN2na2nb2nc2caD0Ev", "na::nb::nc::ca::~ca()");
    }

    Y_UNIT_TEST(Simple2) {
        Check("_ZN2na2nb2caI2cbNS_2nc2cbEFS2_RKS4_EEENS_2ccIT_EENS_2cdIT0_EERKNS_2ceIT1_EE", "na::cc<cb> na::nb::ca<cb, na::nc::cb, cb (na::nc::cb const&)>(na::cd<na::nc::cb>, na::ce<cb (na::nc::cb const&)> const&)");
    }

    Y_UNIT_TEST(List1) {
        Check("_ZNKSt3__110__function6__funcIZN4DLCL8DLFutureIP15AnalysenManagerE3setINS_8functionIFS5_vEEEJEEEvT_DpOT0_EUlvE_NS_9allocatorISF_EEFvvEE7__cloneEv", "std::__1::__function::__func<void DLCL::DLFuture<AnalysenManager*>::set<std::__1::function<AnalysenManager* ()> >(std::__1::function<AnalysenManager* ()>)::'lambda'(), std::__1::allocator<void DLCL::DLFuture<AnalysenManager*>::set<std::__1::function<AnalysenManager* ()> >(std::__1::function<AnalysenManager* ()>)::'lambda'()>, void ()>::__clone() const");
    }

    Y_UNIT_TEST(List2) {
        Check("_Z1iIiLi0EMN1d1e1fEFvPcEJEE1aIN1bIN1cIT1_E1gEFvvEE1hEEiS9_", "a<b<c<void (d::e::f::*)(char*)>::g, void ()>::h> i<int, 0, void (d::e::f::*)(char*)>(int, void (d::e::f::*)(char*))");
    }

    Y_UNIT_TEST(Lambda1) {
        Check("_ZUliE_", "'lambda'(int)");
    }

    Y_UNIT_TEST(Lambda2) {
        Check("_ZUlfdE_", "'lambda'(float, double)");
    }

    Y_UNIT_TEST(LambdaGeneric1) {
        Check("_ZUlT_E_", "'lambda'(auto)");
    }

    Y_UNIT_TEST(LambdaGeneric2) {
        Check("_ZUlOT_RT0_E_", "'lambda'(auto&&, auto&)");
    }

    Y_UNIT_TEST(LambdaTemplateParam) {
        Check("_Z1bIZN1cC1EvEUlT_E_EvS1_", "void b<c::c()::'lambda'(auto)>(auto)");
    }

    Y_UNIT_TEST(Difficult1) {
        Check("_ZNSt4__y16vectorIN2na2caINS1_2nb2cbEEENS_9allocatorIS5_EEE6assignIPS5_EENS_9enable_ifIXaasr21__is_forward_iteratorIT_EE5valuesr16is_constructibleIS5_NS_15iterator_traitsISC_E9referenceEEE5valueEvE4typeESC_SC_", "std::__y1::enable_if<(__is_forward_iterator<na::ca<na::nb::cb>*>::value) && (is_constructible<na::ca<na::nb::cb>, std::__y1::iterator_traits<na::ca<na::nb::cb>*>::reference>::value), void>::type std::__y1::vector<na::ca<na::nb::cb>, std::__y1::allocator<na::ca<na::nb::cb> > >::assign<na::ca<na::nb::cb>*>(na::ca<na::nb::cb>*, na::ca<na::nb::cb>*)");
    }

    Y_UNIT_TEST(Difficult2) {
        Check("_ZTSN2na2nb2caINS0_2cbIZZNS_2nc2cc2cd2maERKNS_2ceINS_2nd2cfINS_2ne2nf2cgEEEEERKNS6_INS9_2chEEEENKUlPT_E_clINSA_2ciEEEDaSL_EUlvE_EEFvvESR_EE", "typeinfo name for na::nb::ca<na::nb::cb<auto na::nc::cc::cd::ma(na::ce<na::nd::cf<na::ne::nf::cg> > const&, na::ce<na::ne::ch> const&)::'lambda'(auto*)::operator()<na::ne::nf::ci>(auto*) const::'lambda'()>, void (), void ()>");
    }

    Y_UNIT_TEST(Difficult3) {
        Check("_ZZN2na2nb2ca2maINS0_2cbEEEvRK2ccMT_FvRK2cdR2ceR2cfENS0_2cgEENKUlOS7_OT0_OT1_E_clISA_SC_SE_EEDaSI_SK_SM_", "auto void na::nb::ca::ma<na::nb::cb>(cc const&, void (na::nb::cb::*)(cd const&, ce&, cf&), na::nb::cg)::'lambda'(na::nb::cb&&, auto&&, auto&&)::operator()<cd const&, ce&, cf&>(na::nb::cb&&, auto&&, auto&&) const");
    }
}
