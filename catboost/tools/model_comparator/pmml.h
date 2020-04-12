#pragma once

#include <contrib/libs/pugixml/pugixml.hpp>

#include <library/cpp/string_utils/ztstrbuf/ztstrbuf.h>

#include <util/generic/ptr.h>


namespace NCB {

    // to have a type distinct from pugi::xml_document
    struct TPmmlModel {
        // THolder because TPmmlModel has to be moveable, but pugi::xml_document is not
        THolder<pugi::xml_document> Model;

    public:
        explicit TPmmlModel(TZtStringBuf fileName);
    };
}
