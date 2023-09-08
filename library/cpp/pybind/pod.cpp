#include "pod.h"

namespace NPyBind {
    class TPODAttrGetter: public TBaseAttrGetter<TPOD> {
    public:
        bool GetAttr(PyObject*, const TPOD& self, const TString& attr, PyObject*& res) const override {
            res = self.GetAttr(attr.c_str());
            return res != nullptr;
        }
    };

    TPODTraits::TPODTraits()
        : MyParent("TPOD", "simple struct")
    {
        AddGetter("", new TPODAttrGetter);
    }

}
