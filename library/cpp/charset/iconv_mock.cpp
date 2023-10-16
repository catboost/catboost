#include "iconv.h"

using namespace NICONVPrivate;

TDescriptor::TDescriptor(const char* from, const char* to)
    : Descriptor_((void*)(-1))
    , From_(from)
    , To_(to)
{

}

TDescriptor::~TDescriptor() {
}

size_t NICONVPrivate::RecodeImpl(const TDescriptor& /*descriptor*/, const char* /*in*/, char* /*out*/, size_t /*inSize*/, size_t /*outSize*/, size_t& /*read*/, size_t& /*written*/) {
    ythrow yexception() << "Iconv functionality disabled";
}

void NICONVPrivate::DoRecode(const TDescriptor& /*descriptor*/, const char* /*in*/, char* /*out*/, size_t /*inSize*/, size_t /*outSize*/, size_t& /*read*/, size_t& /*written*/) {
    ythrow yexception() << "Iconv functionality disabled";
}

RECODE_RESULT NICONVPrivate::DoRecodeNoThrow(const TDescriptor& /*descriptor*/, const char* /*in*/, char* /*out*/, size_t /*inSize*/, size_t /*outSize*/, size_t& /*read*/, size_t& /*written*/) {
    return RECODE_ERROR;
}
