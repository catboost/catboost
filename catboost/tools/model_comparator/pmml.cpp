#include "pmml.h"

#include "decl.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/yexception.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/string/builder.h>
#include <util/string/vector.h>
#include <util/system/compiler.h>

#include <cstring>


using namespace NCB;


struct TXmlComparisonContext {

    THashSet<TString> IgnoredPaths;
    THashMap<TString, THashSet<TString>> IgnoredAttributes; // path -> attribute
};


static TStringBuf GetNodeTypeAsStringBuf(pugi::xml_node_type nodeType) {
    switch (nodeType) {
#define XML_NODE_TYPE_CASE(name) \
        case pugi::name: \
            return TStringBuf(#name);

        XML_NODE_TYPE_CASE(node_null)
        XML_NODE_TYPE_CASE(node_document)
        XML_NODE_TYPE_CASE(node_element)
        XML_NODE_TYPE_CASE(node_pcdata)
        XML_NODE_TYPE_CASE(node_cdata)
        XML_NODE_TYPE_CASE(node_comment)
        XML_NODE_TYPE_CASE(node_pi)
        XML_NODE_TYPE_CASE(node_declaration)
        XML_NODE_TYPE_CASE(node_doctype)

#undef XML_NODE_TYPE_CASE
    }
    Y_UNREACHABLE();
}


static bool CompareAttributes(
    const pugi::xml_node& node1,
    const pugi::xml_node& node2,
    const TXmlComparisonContext& comparisonContext,
    TString* diffString) {

    const auto* ignoredAttributesMap = comparisonContext.IgnoredAttributes.FindPtr(node1.path());

    size_t attrIdx = 0; // for output
    auto attrIter1 = node1.attributes_begin();
    auto attrIter2 = node2.attributes_begin();

    while (true) {
        if (attrIter1 == node1.attributes_end()) {
            if (attrIter2 == node2.attributes_end()) {
                return true;
            } else {
                TStringOutput out(*diffString);
                out << "node " << node1.path() << " has attribute \"" << attrIter2->name()
                    << "\" in model2 but not in model1";
                return false;
            }
        } else {
            if (attrIter2 == node2.attributes_end()) {
                TStringOutput out(*diffString);
                out << "node " << node1.path() << " has attribute \"" << attrIter1->name()
                    << "\" in model1 but not in model2";
                return false;
            }

            if (std::strcmp(attrIter1->name(), attrIter2->name())) {
                TStringOutput out(*diffString);
                out << "node " << node1.path() << " has different attributes at index " << attrIdx
                    << ": model1 has \"" << attrIter1->name() << "\", model2 has \"" << attrIter2->name()
                    << "\"";
                return false;
            }

            if (!ignoredAttributesMap || !ignoredAttributesMap->contains(attrIter1->name())) {
                if (std::strcmp(attrIter1->value(), attrIter2->value())) {
                    TStringOutput out(*diffString);
                    out << "node " << node1.path() << ": attribute \"" << attrIter1->name()
                        << "\" has different values: model1 has \"" << attrIter1->value()
                        << "\", model2 has \"" << attrIter2->value() << "\"";
                    return false;
                }
            }

            ++attrIdx;
            ++attrIter1;
            ++attrIter2;
        }
    }

    Y_UNREACHABLE();
    return true; // make compiler happy
}


static bool CompareNodes(
    const pugi::xml_node& node1,
    const pugi::xml_node& node2,
    const TXmlComparisonContext& comparisonContext,
    TString* diffString) {

    if (node1.type() == pugi::node_document) {
        CB_ENSURE(node2.type() == pugi::node_document, "Document node is compared with non-document node");
    } else {
        if (std::strcmp(node1.name(), node2.name())) {
            TStringOutput out(*diffString);
            out << "models has nodes with different names at path " << node1.parent().path()
                << " : model1 has node " << node1.name() << ", model2 has node " << node2.name();
            return false;
        }

        if (node1.type() != node2.type()) {
            TStringOutput out(*diffString);
            out << "models has nodes with different types at path " << node1.path()
                << " : model1 has type " << GetNodeTypeAsStringBuf(node1.type())
                << ", model2 has type " << GetNodeTypeAsStringBuf(node2.type());
            return false;
        }

        if (node1.type() != pugi::node_comment) {
            if (std::strcmp(node1.value(), node2.value())) {
                TStringOutput out(*diffString);
                out << "models has nodes with different values at path " << node1.path()
                    << " : model1 has node with value \"" << node1.value()
                    << "\", model2 has node with value \"" << node2.value() << "\"";
                return false;
            }
        }
    }

    if (!CompareAttributes(node1, node2, comparisonContext, diffString)) {
        return false;
    }

    auto childNodeIter1 = node1.begin();
    auto childNodeIter2 = node2.begin();

    while (true) {
        if (childNodeIter1 == node1.end()) {
            if (childNodeIter2 == node2.end()) {
                return true;
            } else {
                TStringOutput out(*diffString);
                out << "model2 has node " << childNodeIter2->path() << "but model1 has not";
                return false;
            }
        } else {
            if (childNodeIter2 == node2.end()) {
                TStringOutput out(*diffString);
                out << "model1 has node " << childNodeIter1->path() << "but model2 has not";
                return false;
            }

            if (!comparisonContext.IgnoredPaths.contains(childNodeIter1->path())) {
                if (!CompareNodes(*childNodeIter1, *childNodeIter2, comparisonContext, diffString)) {
                    return false;
                }
            }

            ++childNodeIter1;
            ++childNodeIter2;
        }
    }

    Y_UNREACHABLE();
    return true; // make compiler happy
}


namespace NCB {

    TPmmlModel::TPmmlModel(TZtStringBuf fileName)
        : Model(MakeHolder<pugi::xml_document>())
    {
        auto parseResult = Model->load_file(fileName.c_str());
        CB_ENSURE(parseResult, "Failed to load_file " << fileName << ": " << parseResult.description());
    }

    template <>
    TMaybe<TPmmlModel> TryLoadModel<TPmmlModel>(TStringBuf filePath) {
        TMaybe<TPmmlModel> result;
        try {
            result.ConstructInPlace(TString(filePath));
        } catch (yexception& e) {
            return Nothing();
        }
        if (std::strcmp(result->Model->first_child().name(), "PMML")) {
            return Nothing();
        }
        return result;
    }

    template <>
    bool CompareModels(const TPmmlModel& model1, const TPmmlModel& model2, double, TString* diffString) {
        TXmlComparisonContext comparisonContext;
        comparisonContext.IgnoredPaths.emplace("/PMML/Header/Timestamp");
        comparisonContext.IgnoredAttributes.emplace("/PMML/Header/Application", THashSet<TString>{"version"});

        return CompareNodes(model1.Model->first_child(), model2.Model->first_child(), comparisonContext, diffString);
    }

}

