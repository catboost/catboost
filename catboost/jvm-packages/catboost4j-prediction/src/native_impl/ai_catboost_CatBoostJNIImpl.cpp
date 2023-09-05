#include "ai_catboost_CatBoostJNIImpl.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/output_file_options.h>

#include <util/generic/cast.h>
#include <util/generic/maybe.h>
#include <util/generic/noncopyable.h>
#include <util/generic/scope.h>
#include <util/generic/singleton.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/xrange.h>
#include <util/stream/labeled.h>
#include <util/system/platform.h>

#include <exception>

#if defined (_32_)
    #error "sorry, not expected to work on 32-bit platform"
#endif


#define Y_BEGIN_JNI_API_CALL() \
    try {

#define Y_END_JNI_API_CALL()                   \
    } catch (const std::exception& exc) {      \
        return jenv->NewStringUTF(exc.what()); \
    }                                          \
                                               \
    return nullptr;

// `int` and `jint` may be different types, e.g. on windows `jint` is a typedef for `long` and `int`
// and `long` are different types, but what we really care about is `int` and `jint` binary
// compatibility, so instead of checking if `jint` is really an `int` we check if `jint` and `int`
// have the same type (so it will be safe to do `reinterpret_cast` for between them).
static_assert(sizeof(jint) == sizeof(int), "jint and int have different sizes");
static_assert(sizeof(jbyte) == sizeof(char), "jbyte and char size are not the same");
static_assert(sizeof(jlong) == sizeof(void*), "handle size doesn't match pointer size");
static_assert(std::is_same<jfloat, float>::value, "jfloat and float are not the same type");
static_assert(std::is_same<jdouble, double>::value, "jdouble and double are no the same type");

namespace {
    using TFullModelPtr = TFullModel*;
    using TConstFullModelPtr = const TFullModel*;
}

static TFullModelPtr ToFullModelPtr(const jlong handle) {
    return reinterpret_cast<TFullModelPtr>(handle);
}

static TConstFullModelPtr ToConstFullModelPtr(const jlong handle) {
    return reinterpret_cast<TConstFullModelPtr>(handle);
}

static jlong ToHandle(const void* ptr) {
    return reinterpret_cast<jlong>(ptr);
}

// Note you can't use NewStringUTF/GetStringUTFChars for anything except ASCII
// The "UTF" in the names is "modified UTF-8" - see
// https://docs.oracle.com/javase/7/docs/technotes/guides/jni/spec/types.html#wp16542
// for details.

static jstring StringToJavaUTF8(JNIEnv* const jenv, const TString& string) {
    auto byteArray = jenv->NewByteArray(string.size());
    CB_ENSURE(byteArray, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, byteArray) {
        jenv->DeleteLocalRef(byteArray);
    };

    jenv->SetByteArrayRegion(byteArray, 0, string.size(), (const jbyte*) string.c_str());

    // The ritual below calls `new String(byteArray, "UTF-8")`.
    auto cls = jenv->FindClass("java/lang/String");
    CB_ENSURE(cls, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, cls) {
        jenv->DeleteLocalRef(cls);
    };

    auto ctor = jenv->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    CB_ENSURE(ctor, "OutOfMemoryError");

    auto encoding = jenv->NewStringUTF("UTF-8");
    CB_ENSURE(encoding, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, encoding) {
        jenv->DeleteLocalRef(encoding);
    };

    auto converted = (jstring)jenv->NewObject(cls, ctor, byteArray, encoding);
    CB_ENSURE(converted, "OutOfMemoryError");
    return converted;
}

static jbyteArray JavaToStringUTF8(JNIEnv* jenv, const jstring& str) {
    // Ritual: call `str.getBytes("UTF-8")`
    jclass stringClass = jenv->GetObjectClass(str);
    CB_ENSURE(stringClass, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, stringClass) {
        jenv->DeleteLocalRef(stringClass);
    };

    const jmethodID getBytes = jenv->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    CB_ENSURE(getBytes, "OutOfMemoryError");

    auto encoding = jenv->NewStringUTF("UTF-8");
    CB_ENSURE(encoding, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, encoding) {
        jenv->DeleteLocalRef(encoding);
    };

    jbyteArray bytes = (jbyteArray) jenv->CallObjectMethod(str, getBytes, encoding);
    CB_ENSURE(bytes, "OutOfMemoryError");
    return bytes;
}

class TJVMFloatArrayAsArrayRef : private TMoveOnly {
public:
    TJVMFloatArrayAsArrayRef(JNIEnv* jenv, jfloatArray floatArray, size_t numElements)
        : JEnv(jenv)
        , FloatArray(floatArray)
    {
        jboolean isCopy = JNI_FALSE;
        jfloat* floatArrayRaw = jenv->GetFloatArrayElements(floatArray, &isCopy);
        CB_ENSURE(floatArrayRaw, "OutOfMemoryError");
        Data = MakeArrayRef(floatArrayRaw, numElements);
    }

    TJVMFloatArrayAsArrayRef(JNIEnv* jenv, jfloatArray floatArray)
         : TJVMFloatArrayAsArrayRef(jenv, floatArray, jenv->GetArrayLength(floatArray))
    {}

    TJVMFloatArrayAsArrayRef(TJVMFloatArrayAsArrayRef&& rhs)
        : JEnv(rhs.JEnv)
        , FloatArray(rhs.FloatArray)
        , Data(rhs.Data)
    {
        rhs.JEnv = nullptr;
    }

    ~TJVMFloatArrayAsArrayRef() {
        if (JEnv) {
            JEnv->ReleaseFloatArrayElements(FloatArray, Data.data(), JNI_ABORT);
        }
    }

    TArrayRef<float> Get() const {
        return Data;
    }

private:
    JNIEnv* JEnv;           // if null means this objects has been moved from
    jfloatArray FloatArray;
    TArrayRef<float> Data;
};

class TJVMStringAsStringBuf : private TMoveOnly {
public:
    TJVMStringAsStringBuf(JNIEnv* jenv, jstring string)
        : JEnv(jenv)
        , Utf8Array((jbyteArray)jenv->NewGlobalRef(JavaToStringUTF8(jenv, string)))
    {
        jboolean isCopy = JNI_FALSE;
        jbyte* utf8 = jenv->GetByteArrayElements(Utf8Array, &isCopy);
        CB_ENSURE(utf8, "OutOfMemoryError");
        Data = TStringBuf((const char*) utf8, jenv->GetArrayLength(Utf8Array));
    }

    TJVMStringAsStringBuf(TJVMStringAsStringBuf&& rhs)
        : JEnv(rhs.JEnv)
        , Utf8Array(rhs.Utf8Array)
        , Data(rhs.Data)
    {
        rhs.JEnv = nullptr;
    }

    ~TJVMStringAsStringBuf() {
        if (JEnv) {
            JEnv->ReleaseByteArrayElements(Utf8Array, (signed char*)const_cast<char*>(Data.Data()), JNI_ABORT);
            JEnv->DeleteGlobalRef(Utf8Array);
        }
    }

    TStringBuf Get() const {
        return Data;
    }

private:
    JNIEnv* JEnv;          // if null means this objects has been moved from
    jbyteArray Utf8Array;
    TStringBuf Data;
};

static jint CalcCatFeatureHashJava(JNIEnv* jenv, jstring string) {
    return CalcCatFeatureHash(TJVMStringAsStringBuf(jenv, string).Get());
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostHashCatFeature
  (JNIEnv* jenv, jclass, jstring jcatFeature, jintArray jhash) {
    Y_BEGIN_JNI_API_CALL();

    const auto jhashSize = jenv->GetArrayLength(jhash);
    CB_ENSURE(jhashSize >= 1, "insufficient `hash` size: " LabeledOutput(jhashSize));

    const jint hash = CalcCatFeatureHashJava(jenv, jcatFeature);
    jenv->SetIntArrayRegion(jhash, 0, 1, &hash);

    Y_END_JNI_API_CALL();
}

static void HashCatFeatures(
    JNIEnv* const jenv,
    const jobjectArray jcatFeatures,
    const size_t jcatFeaturesSize,
    const TArrayRef<int> hashes) {

    Y_ASSERT(hashes.size() <= jcatFeaturesSize);

    for (size_t i = 0; i < hashes.size(); ++i) {
        // NOTE: instead of C-style cast `dynamic_cast` should be used, but compiler complains that
        // `_jobject` is not a polymorphic type
        const auto jcatFeature = (jstring)jenv->GetObjectArrayElement(jcatFeatures, i);
        CB_ENSURE(jenv->IsSameObject(jcatFeature, NULL) == JNI_FALSE, "got null array element");
        Y_SCOPE_EXIT(jenv, jcatFeature) {
            jenv->DeleteLocalRef(jcatFeature);
        };
        hashes[i] = CalcCatFeatureHashJava(jenv, jcatFeature);
    }
}

static void GetTextFeatures(
    JNIEnv* const jenv,
    const jobjectArray jtextFeatures,
    const size_t jtextFeaturesSize,
    TVector<TJVMStringAsStringBuf>* textFeaturesStorage,
    TVector<TStringBuf>* textFeatures) {

    textFeaturesStorage->clear();
    textFeatures->clear();

    for (auto i : xrange(jtextFeaturesSize)) {
        // NOTE: instead of C-style cast `dynamic_cast` should be used, but compiler complains that
        // `_jobject` is not a polymorphic type
        const auto jtextFeature = (jstring)jenv->GetObjectArrayElement(jtextFeatures, i);
        CB_ENSURE(jenv->IsSameObject(jtextFeature, NULL) == JNI_FALSE, "got null array element");
        Y_SCOPE_EXIT(jenv, jtextFeature) {
            jenv->DeleteLocalRef(jtextFeature);
        };
        textFeaturesStorage->push_back(TJVMStringAsStringBuf(jenv, jtextFeature));
        textFeatures->push_back(textFeaturesStorage->back().Get());
    }
}

static void GetEmbeddingFeatures(
    JNIEnv* const jenv,
    const jobjectArray jembeddingFeatures,
    const size_t jembeddingFeaturesSize,
    TVector<TJVMFloatArrayAsArrayRef>* embeddingFeaturesStorage,
    TVector<TConstArrayRef<float>>* embeddingFeatures) {

    embeddingFeaturesStorage->clear();
    embeddingFeatures->clear();

    for (auto i : xrange(jembeddingFeaturesSize)) {
        // NOTE: instead of C-style cast `dynamic_cast` should be used, but compiler complains that
        // `_jobject` is not a polymorphic type
        const auto jembeddingFeature = (jfloatArray)jenv->GetObjectArrayElement(jembeddingFeatures, i);
        CB_ENSURE(jenv->IsSameObject(jembeddingFeature, NULL) == JNI_FALSE, "got null array element");
        embeddingFeaturesStorage->push_back(TJVMFloatArrayAsArrayRef(jenv, jembeddingFeature));
        embeddingFeatures->push_back(embeddingFeaturesStorage->back().Get());
    }
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostHashCatFeatures
  (JNIEnv* jenv, jclass, jobjectArray jcatFeatures, jintArray jhashes) {
    Y_BEGIN_JNI_API_CALL();

    const auto jcatFeaturesSize = jenv->GetArrayLength(jcatFeatures);
    const auto jhashesSize = jenv->GetArrayLength(jhashes);

    CB_ENSURE(
        jhashesSize >= jcatFeaturesSize,
        "insufficient `hashes` size: " LabeledOutput(jcatFeaturesSize, jhashesSize));

    const size_t size = jcatFeaturesSize;

    TVector<int> hashes;
    hashes.yresize(size);
    HashCatFeatures(jenv, jcatFeatures, size, hashes);

    jenv->SetIntArrayRegion(jhashes, 0, size, reinterpret_cast<const jint*>(hashes.data()));

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostFreeModel
  (JNIEnv* jenv, jclass, jlong jhandle) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model handle");
    delete model;

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostLoadModelFromFile
  (JNIEnv* jenv, jclass, jstring jmodelPath, jlongArray jhandles, jstring jmodelFormat) {
    Y_BEGIN_JNI_API_CALL();

    const auto modelPathSize = jenv->GetStringUTFLength(jmodelPath);
    const auto* const modelPath = jenv->GetStringUTFChars(jmodelPath, nullptr);
    CB_ENSURE(modelPath, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, jmodelPath, modelPath) {
        jenv->ReleaseStringUTFChars(jmodelPath, modelPath);
    };
    CB_ENSURE(modelPath, "got nullptr modelPath");

    const auto* const modelFormat = jenv->GetStringUTFChars(jmodelFormat, nullptr);
    CB_ENSURE(modelFormat, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, jmodelFormat, modelFormat) {
        jenv->ReleaseStringUTFChars(jmodelFormat, modelFormat);
    };

    EModelType modelType;
    if( !NCatboostOptions::TryGetModelTypeFromExtension(TString(modelFormat), modelType) ) {
        modelType = EModelType::CatboostBinary;
    }

    // TODO(yazevnul): `ReadModel` should return `THolder<TFullModel>` instead of `TFullModel`
    auto model = MakeHolder<TFullModel>();
    *model = ReadModel(TString(modelPath, modelPathSize), modelType);

    const auto handle = ToHandle(model.Get());
    jenv->SetLongArrayRegion(jhandles, 0, 1, &handle);

    // model is now owned by Java object
    Y_UNUSED(model.Release());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostLoadModelFromArray
  (JNIEnv* jenv, jclass, jbyteArray jdata, jlongArray jhandles, jstring jmodelFormat) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const data = jenv->GetByteArrayElements(jdata, nullptr);
    CB_ENSURE(data, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, jdata, data) {
        jenv->ReleaseByteArrayElements(jdata, const_cast<jbyte*>(data), JNI_ABORT);
    };
    const size_t dataSize = jenv->GetArrayLength(jdata);

    const auto* const modelFormat = jenv->GetStringUTFChars(jmodelFormat, nullptr);
    CB_ENSURE(modelFormat, "OutOfMemoryError");
    Y_SCOPE_EXIT(jenv, jmodelFormat, modelFormat) {
        jenv->ReleaseStringUTFChars(jmodelFormat, modelFormat);
    };

    EModelType modelType;
    if( !NCatboostOptions::TryGetModelTypeFromExtension(TString(modelFormat), modelType) ) {
        modelType = EModelType::CatboostBinary;
    }

    auto model = MakeHolder<TFullModel>();
    *model = ReadModel(data, dataSize, modelType);

    const auto handle = ToHandle(model.Get());
    jenv->SetLongArrayRegion(jhandles, 0, 1, &handle);

    Y_UNUSED(model.Release());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetSupportedEvaluatorTypes
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jevaluatorTypes) {
    Y_BEGIN_JNI_API_CALL();

    Y_UNUSED(jhandle);

    auto supportedEvaluatorTypes = TFullModel::GetSupportedEvaluatorTypes();

    auto evaluatorTypes = jenv->NewObjectArray(
        supportedEvaluatorTypes.size(),
        jenv->FindClass("java/lang/String"),
        /*init*/ NULL
    );
    CB_ENSURE(evaluatorTypes, "OutOfMemoryError");

    for (auto i : xrange(supportedEvaluatorTypes.size())) {
        auto jevaluatorType = StringToJavaUTF8(jenv, ToString(supportedEvaluatorTypes[i]));
        jenv->SetObjectArrayElement(evaluatorTypes, static_cast<jsize>(i), jevaluatorType);
    }

    jenv->SetObjectArrayElement(jevaluatorTypes, 0, evaluatorTypes);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelSetEvaluatorType
  (JNIEnv* jenv, jclass, jlong jhandle, jstring jevaluatorType) {
    Y_BEGIN_JNI_API_CALL();

    auto* model = ToFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    TJVMStringAsStringBuf evaluatorTypeStringBuf(jenv, jevaluatorType);
    model->SetEvaluatorType(FromString<EFormulaEvaluatorType>(evaluatorTypeStringBuf.Get()));

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetEvaluatorType
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jevaluatorType) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto evaluatorTypeString = ToString(model->GetEvaluatorType());
    auto jevaluatorTypeString = StringToJavaUTF8(jenv, evaluatorTypeString);
    jenv->SetObjectArrayElement(jevaluatorType, 0, jevaluatorTypeString);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetPredictionDimension
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jpredictionDimension) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint predictionDimension = model->GetDimensionsCount();
    jenv->SetIntArrayRegion(jpredictionDimension, 0, 1, &predictionDimension);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetUsedNumericFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jusedNumericFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint usedNumericFeatureCount = model->GetUsedFloatFeaturesCount();
    jenv->SetIntArrayRegion(jusedNumericFeatureCount, 0, 1, &usedNumericFeatureCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetUsedCategoricalFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jusedCatFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint usedCatFeatureCount = model->GetUsedCatFeaturesCount();
    jenv->SetIntArrayRegion(jusedCatFeatureCount, 0, 1, &usedCatFeatureCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetUsedTextFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jusedTextFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint usedTextFeatureCount = model->GetUsedTextFeaturesCount();
    jenv->SetIntArrayRegion(jusedTextFeatureCount, 0, 1, &usedTextFeatureCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetUsedEmbeddingFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jusedEmbeddingFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint usedEmbeddingFeatureCount = model->GetUsedEmbeddingFeaturesCount();
    jenv->SetIntArrayRegion(jusedEmbeddingFeatureCount, 0, 1, &usedEmbeddingFeatureCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetFlatFeatureVectorExpectedSize
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jfeatureVectorExpectedSize) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint featureVectorExpectedSize = SafeIntegerCast<jint>(
        model->ModelTrees->GetFlatFeatureVectorExpectedSize()
    );
    jenv->SetIntArrayRegion(jfeatureVectorExpectedSize, 0, 1, &featureVectorExpectedSize);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetMetadata
    (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jkeys, jobjectArray jvalues) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto& modelInfo = model->ModelInfo;
    auto keysArray = jenv->NewObjectArray(modelInfo.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(keysArray, "OutOfMemoryError");
    auto valuesArray = jenv->NewObjectArray(modelInfo.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(valuesArray, "OutOfMemoryError");

    int i = 0;
    for (const auto& keyValue : modelInfo) {
        // pair
        auto jkey = StringToJavaUTF8(jenv, keyValue.first);
        jenv->SetObjectArrayElement(keysArray, i, jkey);

        auto jvalue = StringToJavaUTF8(jenv, keyValue.second);
        jenv->SetObjectArrayElement(valuesArray, i, jvalue);

        jenv->DeleteLocalRef(jkey);
        jenv->DeleteLocalRef(jvalue);

        i++;
    }

    jenv->SetObjectArrayElement(jkeys, 0, keysArray);
    jenv->SetObjectArrayElement(jvalues, 0, valuesArray);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetFloatFeatures
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnames, jobjectArray jflat_feature_index, jobjectArray jfeature_index, jobjectArray jhas_nans, jobjectArray jnan_value_treatment)
{
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto& features = model->ModelTrees->GetFloatFeatures();

    auto namesArray = jenv->NewObjectArray(features.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(namesArray, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jnames, 0, namesArray);

    auto flatFeatureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(flatFeatureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jflat_feature_index, 0, flatFeatureIndex);

    auto featureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(featureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jfeature_index, 0, featureIndex);

    auto hasNans = jenv->NewIntArray(features.size());
    CB_ENSURE(hasNans, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jhas_nans, 0, hasNans);

    auto nanValueTreatment = jenv->NewObjectArray(features.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(nanValueTreatment, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jnan_value_treatment, 0, nanValueTreatment);

    int i = 0;
    for (const auto& feature : features) {
        // pair
        auto name = feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId;
        auto jname = StringToJavaUTF8(jenv, name);
        jenv->SetObjectArrayElement(namesArray, i, jname);
        jenv->DeleteLocalRef(jname);

        const jint iFlatIndex = feature.Position.FlatIndex;
        jenv->SetIntArrayRegion(flatFeatureIndex, i, 1, &iFlatIndex);
        const jint iFeatureIndex = feature.Position.Index;
        jenv->SetIntArrayRegion(featureIndex, i, 1, &iFeatureIndex);
        const jint iHasNans = feature.HasNans ? 1 : 0;
        jenv->SetIntArrayRegion(hasNans, i, 1, &iHasNans);

        jstring iNanValueTreatment;
        switch (feature.NanValueTreatment) {
          case TFloatFeature::ENanValueTreatment::AsIs:
            iNanValueTreatment = jenv->NewStringUTF("AsIs");
            break;
          case TFloatFeature::ENanValueTreatment::AsFalse:
            iNanValueTreatment = jenv->NewStringUTF("AsFalse");
            break;
          case TFloatFeature::ENanValueTreatment::AsTrue:
            iNanValueTreatment = jenv->NewStringUTF("AsTrue");
            break;
        }

        CB_ENSURE(iNanValueTreatment, "OutOfMemoryError");

        jenv->SetObjectArrayElement(nanValueTreatment, i, iNanValueTreatment);
        jenv->DeleteLocalRef(iNanValueTreatment);

        i++;
    }

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetCatFeatures
    (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnames, jobjectArray jflat_feature_index, jobjectArray jfeature_index) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto features = model->ModelTrees->GetCatFeatures();

    auto namesArray = jenv->NewObjectArray(features.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(namesArray, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jnames, 0, namesArray);

    auto flatFeatureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(flatFeatureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jflat_feature_index, 0, flatFeatureIndex);

    auto featureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(featureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jfeature_index, 0, featureIndex);

    int i = 0;
    for (const auto& feature : features) {
        // pair
        auto name = feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId;
        auto jname = StringToJavaUTF8(jenv, name);
        jenv->SetObjectArrayElement(namesArray, i, jname);
        jenv->DeleteLocalRef(jname);

        const jint iFlatIndex = feature.Position.FlatIndex;
        jenv->SetIntArrayRegion(flatFeatureIndex, i, 1, &iFlatIndex);
        const jint iIndex = feature.Position.Index;
        jenv->SetIntArrayRegion(featureIndex, i, 1, &iIndex);

        i++;
    }

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetTextFeatures
   (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnames, jobjectArray jflat_feature_index, jobjectArray jfeature_index) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto features = model->ModelTrees->GetTextFeatures();

    auto namesArray = jenv->NewObjectArray(features.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(namesArray, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jnames, 0, namesArray);

    auto flatFeatureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(flatFeatureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jflat_feature_index, 0, flatFeatureIndex);

    auto featureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(featureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jfeature_index, 0, featureIndex);

    int i = 0;
    for (const auto& feature : features) {
        // pair
        auto name = feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId;
        auto jname = StringToJavaUTF8(jenv, name);
        jenv->SetObjectArrayElement(namesArray, i, jname);
        jenv->DeleteLocalRef(jname);

        const jint iFlatIndex = feature.Position.FlatIndex;
        jenv->SetIntArrayRegion(flatFeatureIndex, i, 1, &iFlatIndex);
        const jint iIndex = feature.Position.Index;
        jenv->SetIntArrayRegion(featureIndex, i, 1, &iIndex);

        i++;
    }

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetEmbeddingFeatures
   (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnames, jobjectArray jflat_feature_index, jobjectArray jfeature_index) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const auto features = model->ModelTrees->GetEmbeddingFeatures();

    auto namesArray = jenv->NewObjectArray(features.size(), jenv->FindClass("java/lang/String"), NULL);
    CB_ENSURE(namesArray, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jnames, 0, namesArray);

    auto flatFeatureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(flatFeatureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jflat_feature_index, 0, flatFeatureIndex);

    auto featureIndex = jenv->NewIntArray(features.size());
    CB_ENSURE(featureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jfeature_index, 0, featureIndex);

    int i = 0;
    for (const auto& feature : features) {
        // pair
        auto name = feature.FeatureId == "" ? ToString(feature.Position.FlatIndex) : feature.FeatureId;
        auto jname = StringToJavaUTF8(jenv, name);
        jenv->SetObjectArrayElement(namesArray, i, jname);
        jenv->DeleteLocalRef(jname);

        const jint iFlatIndex = feature.Position.FlatIndex;
        jenv->SetIntArrayRegion(flatFeatureIndex, i, 1, &iFlatIndex);
        const jint iIndex = feature.Position.Index;
        jenv->SetIntArrayRegion(featureIndex, i, 1, &iIndex);

        i++;
    }

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetUsedFeatureIndices
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jflatFeatureIndex) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    auto featureCount = model->GetUsedCatFeaturesCount() + model->GetUsedFloatFeaturesCount() + model->GetUsedTextFeaturesCount() + model->GetUsedEmbeddingFeaturesCount();
    auto flatFeatureIndex = jenv->NewIntArray(featureCount);
    CB_ENSURE(flatFeatureIndex, "OutOfMemoryError");
    jenv->SetObjectArrayElement(jflatFeatureIndex, 0, flatFeatureIndex);

    int index = 0;

    for (const auto& feature : model->ModelTrees->GetEmbeddingFeatures()) {
        if (feature.UsedInModel()) {
            const jint iFlatIndex = feature.Position.FlatIndex;
            jenv->SetIntArrayRegion(flatFeatureIndex, index, 1, &iFlatIndex);
            index++;
        }
    }

    for (const auto& feature : model->ModelTrees->GetTextFeatures()) {
        if (feature.UsedInModel()) {
            const jint iFlatIndex = feature.Position.FlatIndex;
            jenv->SetIntArrayRegion(flatFeatureIndex, index, 1, &iFlatIndex);
            index++;
        }
    }

    for (const auto& feature : model->ModelTrees->GetCatFeatures()) {
        if (feature.UsedInModel()) {
            const jint iFlatIndex = feature.Position.FlatIndex;
            jenv->SetIntArrayRegion(flatFeatureIndex, index, 1, &iFlatIndex);
            index++;
        }
    }

    for (const auto& feature : model->ModelTrees->GetFloatFeatures()) {
        if (feature.UsedInModel()) {
            const jint iFlatIndex = feature.Position.FlatIndex;
            jenv->SetIntArrayRegion(flatFeatureIndex, index, 1, &iFlatIndex);
            index++;
        }
    }

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetTreeCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jtreeCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint treeCount = model->GetTreeCount();
    jenv->SetIntArrayRegion(jtreeCount, 0, 1, &treeCount);

    Y_END_JNI_API_CALL();
}

static size_t GetArraySize(JNIEnv* const jenv, const jarray array) {
    if (jenv->IsSameObject(array, NULL) == JNI_TRUE) {
        return 0;
    }

    return jenv->GetArrayLength(array);
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3F_3Ljava_lang_String_2_3Ljava_lang_String_2_3_3F_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jfloatArray jnumericFeatures, jobjectArray jcatFeatures, jobjectArray jtextFeatures, jobjectArray jembeddingFeatures, jdoubleArray jprediction) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t minTextFeatureCount = model->GetNumTextFeatures();
    const size_t minEmbeddingFeatureCount = model->GetNumEmbeddingFeatures();

    const size_t numericFeatureCount = GetArraySize(jenv, jnumericFeatures);
    const size_t catFeatureCount = GetArraySize(jenv, jcatFeatures);
    const size_t textFeatureCount = GetArraySize(jenv, jtextFeatures);
    const size_t embeddingFeatureCount = GetArraySize(jenv, jembeddingFeatures);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    CB_ENSURE(
        textFeatureCount >= minTextFeatureCount,
        LabeledOutput(textFeatureCount, minTextFeatureCount));

    CB_ENSURE(
        embeddingFeatureCount >= minEmbeddingFeatureCount,
        LabeledOutput(textFeatureCount, minEmbeddingFeatureCount));

    CB_ENSURE(jprediction, "got null prediction");
    const size_t predictionSize = jenv->GetArrayLength(jprediction);
    CB_ENSURE(
        predictionSize >= modelPredictionSize,
        "`prediction` array is too small" LabeledOutput(predictionSize, modelPredictionSize));

    TConstArrayRef<float> numericFeatures;
    if (numericFeatureCount) {
        jfloat* numericFeaturesRaw = jenv->GetFloatArrayElements(jnumericFeatures, nullptr);
        CB_ENSURE(numericFeaturesRaw, "OutOfMemoryError");
        numericFeatures = MakeArrayRef(
            numericFeaturesRaw,
            numericFeatureCount);
    }
    Y_SCOPE_EXIT(jenv, jnumericFeatures, numericFeatures) {
        if (numericFeatures) {
            jenv->ReleaseFloatArrayElements(
                jnumericFeatures,
                const_cast<float*>(numericFeatures.data()),
                JNI_ABORT);
        }
    };

    TVector<int> catFeatures;
    if (catFeatureCount) {
        catFeatures.yresize(catFeatureCount);
        HashCatFeatures(jenv, jcatFeatures, catFeatureCount, catFeatures);
    }

    TVector<TJVMStringAsStringBuf> textFeaturesStrorage;
    TVector<TStringBuf> textFeatures;
    if (textFeatureCount) {
        GetTextFeatures(jenv, jtextFeatures, textFeatureCount, &textFeaturesStrorage, &textFeatures);
    }

    TVector<TJVMFloatArrayAsArrayRef> embeddingFeaturesStorage;
    TVector<TConstArrayRef<float>> embeddingFeatures;
    if (embeddingFeatureCount) {
        GetEmbeddingFeatures(jenv, jembeddingFeatures, embeddingFeatureCount, &embeddingFeaturesStorage, &embeddingFeatures);
    }

    TVector<double> prediction;
    prediction.yresize(modelPredictionSize);

    TConstArrayRef<int> catFeaturesAsArrayRef(catFeatures);
    TConstArrayRef<TConstArrayRef<float>> embeddingFeaturesAsArrayRef(embeddingFeatures);
    model->CalcWithHashedCatAndTextAndEmbeddings(
        MakeArrayRef(&numericFeatures, 1),
        MakeArrayRef(&catFeaturesAsArrayRef, 1),
        MakeArrayRef(&textFeatures, 1),
        MakeArrayRef(&embeddingFeaturesAsArrayRef, 1),
        prediction);

    jenv->SetDoubleArrayRegion(jprediction, 0, modelPredictionSize, prediction.data());

    Y_END_JNI_API_CALL();
}

static size_t GetMatrixColumnCount(JNIEnv* const jenv, const jobjectArray matrix) {
    if (jenv->IsSameObject(matrix, NULL) == JNI_TRUE) {
        return 0;
    }

    const auto rowCount = jenv->GetArrayLength(matrix);
    if (rowCount <= 0) {
        return 0;
    }

    const auto firstRow = (jobjectArray)jenv->GetObjectArrayElement(matrix, 0);
    if (jenv->IsSameObject(firstRow, NULL) == JNI_TRUE) {
        return 0;
    }

    return jenv->GetArrayLength(firstRow);
}

static size_t GetDocumentCount(
    JNIEnv* const jenv,
    jobjectArray jnumericFeaturesMatrix,
    jobjectArray jcatFeaturesMatrix,
    jobjectArray jtextFeaturesMatrix,
    jobjectArray jembeddingFeaturesMatrix) {

    TMaybe<size_t> result;

    for (jobjectArray featuresMatrix : {jnumericFeaturesMatrix, jcatFeaturesMatrix, jtextFeaturesMatrix, jembeddingFeaturesMatrix}) {
        if (jenv->IsSameObject(featuresMatrix, NULL) != JNI_TRUE) {
            size_t documentCount = SafeIntegerCast<size_t>(jenv->GetArrayLength(featuresMatrix));
            if (result) {
                CB_ENSURE(documentCount == *result, "features arrays have different number of objects");
            } else {
                result = documentCount;
            }
        }
    }

    return result.GetOrElse(0);
}


JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3_3F_3_3Ljava_lang_String_2_3_3Ljava_lang_String_2_3_3_3F_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix, jobjectArray jtextFeaturesMatrix, jobjectArray jembeddingFeaturesMatrix, jdoubleArray jpredictions) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const size_t documentCount = GetDocumentCount(jenv, jnumericFeaturesMatrix, jcatFeaturesMatrix, jtextFeaturesMatrix, jembeddingFeaturesMatrix);
    if (documentCount == 0) {
        return 0;
    }

    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t minTextFeatureCount = model->GetNumTextFeatures();
    const size_t minEmbeddingFeatureCount = model->GetNumEmbeddingFeatures();

    const size_t numericFeatureCount = GetMatrixColumnCount(jenv, jnumericFeaturesMatrix);
    const size_t catFeatureCount = GetMatrixColumnCount(jenv, jcatFeaturesMatrix);
    const size_t textFeatureCount = GetMatrixColumnCount(jenv, jtextFeaturesMatrix);
    const size_t embeddingFeatureCount = GetMatrixColumnCount(jenv, jembeddingFeaturesMatrix);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    CB_ENSURE(
        textFeatureCount >= minTextFeatureCount,
        LabeledOutput(textFeatureCount, minTextFeatureCount));

    CB_ENSURE(
        embeddingFeatureCount >= minEmbeddingFeatureCount,
        LabeledOutput(embeddingFeatureCount, minEmbeddingFeatureCount));

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is not sufficient, must be at least document count * prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;

    TVector<int> catFeatureMatrixRowwise;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    TVector<TVector<TJVMStringAsStringBuf>> textFeatureMatrixStorage;
    TVector<TVector<TStringBuf>> textFeatureMatrixRows;

    TVector<TVector<TJVMFloatArrayAsArrayRef>> embeddingFeatureMatrixStorage;
    TVector<TVector<TConstArrayRef<float>>> embeddingFeatureMatrixRows;
    TVector<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatureMatrixRowsAsRefs;

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                JNI_ABORT);
        }
    };

    if (numericFeatureCount) {
        numericFeatureMatrixRowObjects.reserve(documentCount);
        numericFeatureMatrixRows.reserve(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jfloatArray)jenv->GetObjectArrayElement(
                jnumericFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                numericFeatureCount <= rowSize,
                "numeric feature count doesn't match for row " << i << ": "
                LabeledOutput(numericFeatureCount, rowSize));
            numericFeatureMatrixRowObjects.push_back(row);
            numericFeatureMatrixRows.push_back(MakeArrayRef(
                jenv->GetFloatArrayElements(row, nullptr),
                numericFeatureCount));
        }
    }

    if (catFeatureCount) {
        catFeatureMatrixRowwise.yresize(documentCount * catFeatureCount);
        catFeatureMatrixRows.reserve(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jcatFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            Y_SCOPE_EXIT(jenv, row) {
              jenv->DeleteLocalRef(row);
            };
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                catFeatureCount <= rowSize,
                "cat feature count doesn't match for row " << i << ": "
                LabeledOutput(catFeatureCount, rowSize));
            const auto hashes = MakeArrayRef(
                catFeatureMatrixRowwise.data() + i * catFeatureCount,
                catFeatureCount);
            HashCatFeatures(jenv, row, rowSize, hashes);
            catFeatureMatrixRows.push_back(hashes);
        }
    }

    if (textFeatureCount) {
        textFeatureMatrixStorage.resize(documentCount);
        textFeatureMatrixRows.resize(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jtextFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            Y_SCOPE_EXIT(jenv, row) {
              jenv->DeleteLocalRef(row);
            };
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                textFeatureCount <= rowSize,
                "text feature count doesn't match for row " << i << ": "
                LabeledOutput(textFeatureCount, rowSize));
            GetTextFeatures(jenv, row, textFeatureCount, &(textFeatureMatrixStorage[i]), &(textFeatureMatrixRows[i]));
        }
    }

    if (embeddingFeatureCount) {
        embeddingFeatureMatrixStorage.resize(documentCount);
        embeddingFeatureMatrixRows.resize(documentCount);
        embeddingFeatureMatrixRowsAsRefs.resize(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jembeddingFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            Y_SCOPE_EXIT(jenv, row) {
              jenv->DeleteLocalRef(row);
            };
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                embeddingFeatureCount <= rowSize,
                "text feature count doesn't match for row " << i << ": "
                LabeledOutput(embeddingFeatureCount, rowSize));
            GetEmbeddingFeatures(jenv, row, embeddingFeatureCount, &(embeddingFeatureMatrixStorage[i]), &(embeddingFeatureMatrixRows[i]));
            embeddingFeatureMatrixRowsAsRefs[i] = TConstArrayRef<TConstArrayRef<float>>(embeddingFeatureMatrixRows[i]);
        }
    }

    TVector<double> predictions;
    predictions.yresize(documentCount * modelPredictionSize);
    model->CalcWithHashedCatAndTextAndEmbeddings(
        numericFeatureMatrixRows,
        catFeatureMatrixRows,
        textFeatureMatrixRows,
        embeddingFeatureMatrixRowsAsRefs,
        predictions);

    jenv->SetDoubleArrayRegion(jpredictions, 0, predictions.size(), predictions.data());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3F_3I_3Ljava_lang_String_2_3_3F_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jfloatArray jnumericFeatures, jintArray jcatFeatures, jobjectArray jtextFeatures, jobjectArray jembeddingFeatures, jdoubleArray jprediction) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t minTextFeatureCount = model->GetNumTextFeatures();
    const size_t minEmbeddingFeatureCount = model->GetNumEmbeddingFeatures();

    const size_t numericFeatureCount = GetArraySize(jenv, jnumericFeatures);
    const size_t catFeatureCount = GetArraySize(jenv, jcatFeatures);
    const size_t textFeatureCount = GetArraySize(jenv, jtextFeatures);
    const size_t embeddingFeatureCount = GetArraySize(jenv, jembeddingFeatures);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    CB_ENSURE(
        textFeatureCount >= minTextFeatureCount,
        LabeledOutput(textFeatureCount, minTextFeatureCount));

    CB_ENSURE(
        embeddingFeatureCount >= minEmbeddingFeatureCount,
        LabeledOutput(embeddingFeatureCount, minEmbeddingFeatureCount));

    const size_t predictionSize = jenv->GetArrayLength(jprediction);
    CB_ENSURE(
        predictionSize >= modelPredictionSize,
        "`prediction` array is too small" LabeledOutput(predictionSize, modelPredictionSize));

    TConstArrayRef<float> numericFeatures;
    if (numericFeatureCount) {
        jfloat* numericFeaturesRaw = jenv->GetFloatArrayElements(jnumericFeatures, nullptr);
        CB_ENSURE(numericFeaturesRaw, "OutOfMemoryError");
        numericFeatures = MakeArrayRef(
            numericFeaturesRaw,
            numericFeatureCount);
    }
    Y_SCOPE_EXIT(jenv, jnumericFeatures, numericFeatures) {
        if (numericFeatures) {
            jenv->ReleaseFloatArrayElements(
                jnumericFeatures,
                const_cast<float*>(numericFeatures.data()),
                JNI_ABORT);
        }
    };

    TConstArrayRef<int> catFeatures;
    if (catFeatureCount) {
        const int* catFeaturesRaw = reinterpret_cast<const int*>(jenv->GetIntArrayElements(jcatFeatures, nullptr));
        CB_ENSURE(catFeaturesRaw, "OutOfMemoryError");
        catFeatures = MakeArrayRef(
            catFeaturesRaw,
            catFeatureCount);
    }
    Y_SCOPE_EXIT(jenv, jcatFeatures, catFeatures) {
        if (catFeatures) {
            jenv->ReleaseIntArrayElements(
                jcatFeatures,
                const_cast<jint*>(reinterpret_cast<const jint*>(catFeatures.data())),
                JNI_ABORT);
        }
    };

    TVector<TJVMStringAsStringBuf> textFeaturesStrorage;
    TVector<TStringBuf> textFeatures;
    if (textFeatureCount) {
        GetTextFeatures(jenv, jtextFeatures, textFeatureCount, &textFeaturesStrorage, &textFeatures);
    }

    TVector<TJVMFloatArrayAsArrayRef> embeddingFeaturesStorage;
    TVector<TConstArrayRef<float>> embeddingFeatures;
    TConstArrayRef<TConstArrayRef<float>> embeddingFeaturesAsRef;
    if (embeddingFeatureCount) {
        GetEmbeddingFeatures(jenv, jembeddingFeatures, embeddingFeatureCount, &embeddingFeaturesStorage, &embeddingFeatures);
        embeddingFeaturesAsRef = TConstArrayRef<TConstArrayRef<float>>(embeddingFeatures);
    }

    TVector<double> prediction;
    prediction.yresize(modelPredictionSize);
    model->CalcWithHashedCatAndTextAndEmbeddings(
        MakeArrayRef(&numericFeatures, 1),
        MakeArrayRef(&catFeatures, 1),
        MakeArrayRef(&textFeatures, 1),
        MakeArrayRef(&embeddingFeaturesAsRef, 1),
        prediction);

    jenv->SetDoubleArrayRegion(jprediction, 0, modelPredictionSize, prediction.data());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3_3F_3_3I_3_3Ljava_lang_String_2_3_3_3F_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix, jobjectArray jtextFeaturesMatrix, jobjectArray jembeddingFeaturesMatrix, jdoubleArray jpredictions) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const size_t documentCount = GetDocumentCount(jenv, jnumericFeaturesMatrix, jcatFeaturesMatrix, jtextFeaturesMatrix, jembeddingFeaturesMatrix);
    if (documentCount == 0) {
        return 0;
    }

    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t minTextFeatureCount = model->GetNumTextFeatures();
    const size_t minEmbeddingFeatureCount = model->GetNumEmbeddingFeatures();

    const size_t numericFeatureCount = GetMatrixColumnCount(jenv, jnumericFeaturesMatrix);
    const size_t catFeatureCount = GetMatrixColumnCount(jenv, jcatFeaturesMatrix);
    const size_t textFeatureCount = GetMatrixColumnCount(jenv, jtextFeaturesMatrix);
    const size_t embeddingFeatureCount = GetMatrixColumnCount(jenv, jembeddingFeaturesMatrix);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    CB_ENSURE(
        textFeatureCount >= minTextFeatureCount,
        LabeledOutput(textFeatureCount, minTextFeatureCount));

    CB_ENSURE(
        embeddingFeatureCount >= minEmbeddingFeatureCount,
        LabeledOutput(embeddingFeatureCount, minEmbeddingFeatureCount));

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is insufficient, must be at least document count * model prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;

    TVector<jintArray> catFeatureMatrixRowObjects;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    TVector<TVector<TJVMStringAsStringBuf>> textFeatureMatrixStorage;
    TVector<TVector<TStringBuf>> textFeatureMatrixRows;

    TVector<TVector<TJVMFloatArrayAsArrayRef>> embeddingFeatureMatrixStorage;
    TVector<TVector<TConstArrayRef<float>>> embeddingFeatureMatrixRows;
    TVector<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatureMatrixRowsAsRefs;

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                JNI_ABORT);
        }
    };

    if (numericFeatureCount) {
        numericFeatureMatrixRowObjects.reserve(documentCount);
        numericFeatureMatrixRows.reserve(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jfloatArray)jenv->GetObjectArrayElement(jnumericFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                numericFeatureCount <= rowSize,
                "numeric feature count doesn't match for row " << i << ": "
                LabeledOutput(numericFeatureCount, rowSize));
            numericFeatureMatrixRowObjects.push_back(row);
            numericFeatureMatrixRows.push_back(MakeArrayRef(
                jenv->GetFloatArrayElements(row, nullptr),
                numericFeatureCount));
        }
    }

    Y_SCOPE_EXIT(jenv, &catFeatureMatrixRowObjects, &catFeatureMatrixRows) {
        const auto size = catFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseIntArrayElements(
                catFeatureMatrixRowObjects[i],
                const_cast<jint*>(reinterpret_cast<const jint*>(catFeatureMatrixRows[i].data())),
                JNI_ABORT);
        }
    };

    if (catFeatureCount) {
        catFeatureMatrixRowObjects.reserve(documentCount);
        catFeatureMatrixRows.reserve(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jintArray)jenv->GetObjectArrayElement(
                jcatFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                catFeatureCount <= rowSize,
                "cat feature count doesn't match for row " << i << ": "
                LabeledOutput(catFeatureCount, rowSize));
            catFeatureMatrixRowObjects.push_back(row);
            catFeatureMatrixRows.push_back(MakeArrayRef(
                reinterpret_cast<const int*>(jenv->GetIntArrayElements(row, nullptr)),
                catFeatureCount));
        }
    }

    if (textFeatureCount) {
        textFeatureMatrixStorage.resize(documentCount);
        textFeatureMatrixRows.resize(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jtextFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            Y_SCOPE_EXIT(jenv, row) {
              jenv->DeleteLocalRef(row);
            };
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                textFeatureCount <= rowSize,
                "text feature count doesn't match for row " << i << ": "
                LabeledOutput(textFeatureCount, rowSize));
            GetTextFeatures(jenv, row, textFeatureCount, &(textFeatureMatrixStorage[i]), &(textFeatureMatrixRows[i]));
        }
    }

    if (embeddingFeatureCount) {
        embeddingFeatureMatrixStorage.resize(documentCount);
        embeddingFeatureMatrixRows.resize(documentCount);
        embeddingFeatureMatrixRowsAsRefs.resize(documentCount);
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jembeddingFeaturesMatrix, i);
            CB_ENSURE(jenv->IsSameObject(row, NULL) == JNI_FALSE, "got null row");
            Y_SCOPE_EXIT(jenv, row) {
              jenv->DeleteLocalRef(row);
            };
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                embeddingFeatureCount <= rowSize,
                "text feature count doesn't match for row " << i << ": "
                LabeledOutput(embeddingFeatureCount, rowSize));
            GetEmbeddingFeatures(jenv, row, embeddingFeatureCount, &(embeddingFeatureMatrixStorage[i]), &(embeddingFeatureMatrixRows[i]));
            embeddingFeatureMatrixRowsAsRefs[i] = TConstArrayRef<TConstArrayRef<float>>(embeddingFeatureMatrixRows[i]);
        }
    }

    TVector<double> predictions;
    predictions.yresize(documentCount * modelPredictionSize);
    model->CalcWithHashedCatAndTextAndEmbeddings(
        numericFeatureMatrixRows,
        catFeatureMatrixRows,
        textFeatureMatrixRows,
        embeddingFeatureMatrixRowsAsRefs,
        predictions);

    jenv->SetDoubleArrayRegion(jpredictions, 0, predictions.size(), predictions.data());

    Y_END_JNI_API_CALL();
}

#undef Y_BEGIN_JNI_API_CALL
#undef Y_END_JNI_API_CALL
