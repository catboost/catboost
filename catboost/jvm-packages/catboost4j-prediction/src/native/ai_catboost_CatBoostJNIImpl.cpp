#include "ai_catboost_CatBoostJNIImpl.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>

#include <util/generic/scope.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/stream/labeled.h>
#include <util/system/platform.h>

#include <exception>

#if defined (_32_)
    #error "sorry, not expected to work on 32-bit platform"
#endif

// TODO(yazevnul): current implementation invokes `Get<PrimitiveType>ArrayRegion` with `mode=0`
// which which asks JRE to make a copy of array [1] and then we invoke
// `Release<PrimitiveType>ArrayElements` with `mode=0` which asks JRE to copy elements back and free
// the buffer. In most of the cases we have no need to copy elements back (because we don't change
// them), and in some cases we can use that and avoid alocation of our own arrays.
//
// [1] https://docs.oracle.com/javase/6/docs/technotes/guides/jni/spec/functions.html

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

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostHashCatFeature
  (JNIEnv* jenv, jclass, jstring jcatFeature, jintArray jhash) {
    Y_BEGIN_JNI_API_CALL();

    const auto jhashSize = jenv->GetArrayLength(jhash);
    CB_ENSURE(jhashSize >= 1, "insufficient `hash` size: " LabeledOutput(jhashSize));

    const auto catFeatureSize = jenv->GetStringUTFLength(jcatFeature);
    const auto* const catFeature = jenv->GetStringUTFChars(jcatFeature, nullptr);
    Y_SCOPE_EXIT(jenv, jcatFeature, catFeature) {
        jenv->ReleaseStringUTFChars(jcatFeature, catFeature);
    };

    const jint hash = CalcCatFeatureHash(TStringBuf(catFeature, catFeatureSize));
    jenv->SetIntArrayRegion(jhash, 0, 1, &hash);

    Y_END_JNI_API_CALL();
}

static void HashCatFeatures(
    JNIEnv* const jenv,
    const jobjectArray jcatFeatures,
    const size_t jcatFeaturesSize,
    const TArrayRef<int> hashes) {

    Y_ASSERT(hashes.size() == jcatFeaturesSize);

    for (size_t i = 0; i < hashes.size(); ++i) {
        // NOTE: instead of C-style cast `dynamic_cast` should be used, but compiler complains that
        // `_jobject` is not a polymorphic type
        const auto jcatFeature = (jstring)jenv->GetObjectArrayElement(jcatFeatures, i);
        const auto catFeatureSize = jenv->GetStringUTFLength(jcatFeature);
        const auto* const catFeature = jenv->GetStringUTFChars(jcatFeature, nullptr);
        Y_SCOPE_EXIT(jenv, jcatFeature, catFeature) {
            jenv->ReleaseStringUTFChars(jcatFeature, catFeature);
        };

        hashes[i] = CalcCatFeatureHash(TStringBuf(catFeature, catFeatureSize));
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
  (JNIEnv* jenv, jclass, jstring jmodelPath, jlongArray jhandles) {
    Y_BEGIN_JNI_API_CALL();

    const auto modelPathSize = jenv->GetStringUTFLength(jmodelPath);
    const auto* const modelPath = jenv->GetStringUTFChars(jmodelPath, nullptr);
    Y_SCOPE_EXIT(jenv, jmodelPath, modelPath) {
        jenv->ReleaseStringUTFChars(jmodelPath, modelPath);
    };
    CB_ENSURE(modelPath, "got nullptr modelPath");

    // TODO(yazevnul): `ReadModel` should return `THolder<TFullModel>` instead of `TFullModel`
    auto model = MakeHolder<TFullModel>();
    *model = ReadModel(TString(modelPath, modelPathSize));

    const auto handle = ToHandle(model.Get());
    jenv->SetLongArrayRegion(jhandles, 0, 1, &handle);

    // model is now owned by Java object
    Y_UNUSED(model.Release());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostLoadModelFromArray
  (JNIEnv* jenv, jclass, jbyteArray jdata, jlongArray jhandles) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const data = jenv->GetByteArrayElements(jdata, nullptr);
    Y_SCOPE_EXIT(jenv, jdata, data) {
        jenv->ReleaseByteArrayElements(jdata, const_cast<jbyte*>(data), 0);
    };
    const size_t dataSize = jenv->GetArrayLength(jdata);

    auto model = MakeHolder<TFullModel>();
    *model = ReadModel(data, dataSize);

    const auto handle = ToHandle(model.Get());
    jenv->SetLongArrayRegion(jhandles, 0, 1, &handle);

    Y_UNUSED(model.Release());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetPredictionDimension
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jpredictionDimension) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint predictionDimension = model->ObliviousTrees.ApproxDimension;
    jenv->SetIntArrayRegion(jpredictionDimension, 0, 1, &predictionDimension);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetNumericFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jnumericFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint numericFeatureCount = model->GetNumFloatFeatures();
    jenv->SetIntArrayRegion(jnumericFeatureCount, 0, 1, &numericFeatureCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetCategoricalFeatureCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jcatFeatureCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint catFeatureCount = model->GetNumCatFeatures();
    jenv->SetIntArrayRegion(jcatFeatureCount, 0, 1, &catFeatureCount);

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

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3F_3Ljava_lang_String_2_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jfloatArray jnumericFeatures, jobjectArray jcatFeatures, jdoubleArray jprediction) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->ObliviousTrees.ApproxDimension;

    const size_t predictionSize = jenv->GetArrayLength(jprediction);
    CB_ENSURE(
        predictionSize >= modelPredictionSize,
        "`prediction` array is too small" LabeledOutput(predictionSize, modelPredictionSize));

    const bool hasNumericFeatures = !jenv->IsSameObject(jnumericFeatures, NULL);
    const bool hasCatFeatures = !jenv->IsSameObject(jcatFeatures, NULL);

    CB_ENSURE(
        hasNumericFeatures || hasCatFeatures,
        "either numeric or categoric features must be provided: "
        LabeledOutput(hasNumericFeatures, hasCatFeatures));

    const size_t numericFeatureCount = hasNumericFeatures
        ? jenv->GetArrayLength(jnumericFeatures)
        : 0;
    const size_t catFeatureCount = hasCatFeatures
        ? jenv->GetArrayLength(jcatFeatures)
        : 0;

    const TConstArrayRef<float> numericFeatures = hasNumericFeatures
        ? MakeArrayRef(
            jenv->GetFloatArrayElements(jnumericFeatures, nullptr),
            numericFeatureCount)
        : TConstArrayRef<float>();
    Y_SCOPE_EXIT(jenv, jnumericFeatures, numericFeatures) {
        if (numericFeatures) {
            jenv->ReleaseFloatArrayElements(
                jnumericFeatures,
                const_cast<float*>(numericFeatures.data()),
                0);
        }
    };

    TVector<int> catFeatures;
    if (hasCatFeatures) {
        catFeatures.yresize(catFeatureCount);
        HashCatFeatures(jenv, jcatFeatures, catFeatureCount, catFeatures);
    }

    TVector<double> prediction;
    prediction.yresize(modelPredictionSize);
    model->Calc(numericFeatures, catFeatures, prediction);

    jenv->SetDoubleArrayRegion(jprediction, 0, modelPredictionSize, prediction.data());

    Y_END_JNI_API_CALL();
}

static size_t GetMatrixSecondDimension(JNIEnv* const jenv, const jobjectArray jmatrix) {
    const auto row = (jobjectArray)jenv->GetObjectArrayElement(jmatrix, 0);
    const size_t size = jenv->GetArrayLength(row);
    return size;
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3_3F_3_3Ljava_lang_String_2_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix, jdoubleArray jpredictions) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->ObliviousTrees.ApproxDimension;

    const bool hasNumericFeatures = !jenv->IsSameObject(jnumericFeaturesMatrix, NULL);
    const bool hasCatFeatures = !jenv->IsSameObject(jcatFeaturesMatrix, NULL);

    CB_ENSURE(
        hasNumericFeatures || hasCatFeatures,
        "either numeric or categoric features must be provided: "
        LabeledOutput(hasNumericFeatures, hasCatFeatures));

    const size_t numericFeatureMatrixFirstDimension = hasNumericFeatures
        ? jenv->GetArrayLength(jnumericFeaturesMatrix)
        : 0;
    const size_t catFeatureMatrixFirstDimension = hasCatFeatures
        ? jenv->GetArrayLength(jcatFeaturesMatrix)
        : 0;

    CB_ENSURE(
        !(hasNumericFeatures && hasCatFeatures) ||
        numericFeatureMatrixFirstDimension == catFeatureMatrixFirstDimension,
        "numeric features array size doesn't match cat features array size: "
        LabeledOutput(numericFeatureMatrixFirstDimension, catFeatureMatrixFirstDimension));

    const auto documentCount = hasNumericFeatures
        ? numericFeatureMatrixFirstDimension
        : catFeatureMatrixFirstDimension;

    if (documentCount == 0) {
        return 0;
    }

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is not sufficient, must be at least document count * prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    const size_t numericFeatureCount = hasNumericFeatures
        ? GetMatrixSecondDimension(jenv, jnumericFeaturesMatrix)
        : 0;
    const size_t catFeatureCount = hasCatFeatures
        ? GetMatrixSecondDimension(jenv, jcatFeaturesMatrix)
        : 0;

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;
    TVector<int> catFeatureMatrixRowwise;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    if (hasNumericFeatures) {
        numericFeatureMatrixRowObjects.reserve(documentCount);
        numericFeatureMatrixRows.reserve(documentCount);
    }

    if (hasCatFeatures) {
        catFeatureMatrixRowwise.reserve(documentCount * numericFeatureCount);
        catFeatureMatrixRows.reserve(documentCount);
    }

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                0);
        }
    };

    if (hasNumericFeatures) {
        for (size_t i = 0; i < documentCount; ++i) {
            numericFeatureMatrixRowObjects.push_back((jfloatArray)jenv->GetObjectArrayElement(
                jnumericFeaturesMatrix, i));
            const size_t rowSize = jenv->GetArrayLength(numericFeatureMatrixRowObjects.back());
            CB_ENSURE(
                numericFeatureCount == rowSize,
                "numeric feature count doesn't match for row " << i << ": "
                LabeledOutput(numericFeatureCount, rowSize));
            numericFeatureMatrixRows.push_back(MakeArrayRef(
                jenv->GetFloatArrayElements(numericFeatureMatrixRowObjects.back(), nullptr),
                numericFeatureCount));
        }
    }

    if (hasCatFeatures) {
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jcatFeaturesMatrix, i);
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                catFeatureCount == rowSize,
                "cat feature count doesn't match for row " << i << ": "
                LabeledOutput(catFeatureCount, rowSize));
            const auto hashes = MakeArrayRef(
                catFeatureMatrixRowwise.data() + i * catFeatureCount,
                catFeatureCount);
            HashCatFeatures(jenv, row, rowSize, hashes);
            catFeatureMatrixRows.push_back(hashes);
        }
    }

    TVector<double> predictions;
    predictions.yresize(documentCount * modelPredictionSize);
    model->Calc(numericFeatureMatrixRows, catFeatureMatrixRows, predictions);

    jenv->SetDoubleArrayRegion(jpredictions, 0, predictions.size(), predictions.data());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3F_3I_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jfloatArray jnumericFeatures, jintArray jcatFeatures, jdoubleArray jprediction) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->ObliviousTrees.ApproxDimension;

    const size_t predictionSize = jenv->GetArrayLength(jprediction);
    CB_ENSURE(
        predictionSize >= modelPredictionSize,
        "`prediction` array is too small" LabeledOutput(predictionSize, modelPredictionSize));

    const bool hasNumericFeatures = !jenv->IsSameObject(jnumericFeatures, NULL);
    const bool hasCatFeatures = !jenv->IsSameObject(jcatFeatures, NULL);

    CB_ENSURE(
        hasNumericFeatures || hasCatFeatures,
        "either numeric or categoric feature hashes must be provided;"
        LabeledOutput(hasNumericFeatures, hasCatFeatures));

    const size_t numericFeatureCount = hasNumericFeatures
        ? jenv->GetArrayLength(jnumericFeatures)
        : 0;
    const size_t catFeaturesCount = hasCatFeatures
        ? jenv->GetArrayLength(jcatFeatures)
        : 0;

    const TConstArrayRef<float> numericFeatures = hasNumericFeatures
        ? MakeArrayRef(
            jenv->GetFloatArrayElements(jnumericFeatures, nullptr),
            numericFeatureCount)
        : TConstArrayRef<float>();
    Y_SCOPE_EXIT(jenv, jnumericFeatures, numericFeatures) {
        if (numericFeatures) {
            jenv->ReleaseFloatArrayElements(
                jnumericFeatures,
                const_cast<float*>(numericFeatures.data()),
                0);
        }
    };
    const TConstArrayRef<int> catFeatures = hasCatFeatures
        ? MakeArrayRef(
            reinterpret_cast<const int*>(jenv->GetIntArrayElements(jcatFeatures, nullptr)),
            catFeaturesCount)
        : TConstArrayRef<int>();
    Y_SCOPE_EXIT(jenv, jcatFeatures, catFeatures) {
        if (catFeatures) {
            jenv->ReleaseIntArrayElements(
                jcatFeatures,
                const_cast<jint*>(reinterpret_cast<const jint*>(catFeatures.data())),
                0);
        }
    };

    TVector<double> prediction;
    prediction.yresize(modelPredictionSize);
    model->Calc(numericFeatures, catFeatures, prediction);

    jenv->SetDoubleArrayRegion(jprediction, 0, modelPredictionSize, prediction.data());

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3_3F_3_3I_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix, jdoubleArray jpredictions) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->ObliviousTrees.ApproxDimension;

    const bool hasNumericFeatures = !jenv->IsSameObject(jnumericFeaturesMatrix, NULL);
    const bool hasCatFeatures = !jenv->IsSameObject(jcatFeaturesMatrix, NULL);

    CB_ENSURE(
        hasNumericFeatures || hasCatFeatures,
        "either numeric or categoric feature hashes must be provided: "
        LabeledOutput(hasNumericFeatures, hasCatFeatures));

    const size_t numericFeatureMatrixFirstDimension = hasNumericFeatures
        ? jenv->GetArrayLength(jnumericFeaturesMatrix)
        : 0;
    const size_t catFeaturesMatrixFirstDimension = hasCatFeatures
        ? jenv->GetArrayLength(jcatFeaturesMatrix)
        : 0;

    CB_ENSURE(
        !(hasNumericFeatures && hasCatFeatures) ||
        numericFeatureMatrixFirstDimension == catFeaturesMatrixFirstDimension,
        "numeric features array size doesn't match cat features array size: "
        LabeledOutput(numericFeatureMatrixFirstDimension, catFeaturesMatrixFirstDimension));

    const auto documentCount = hasNumericFeatures
        ? numericFeatureMatrixFirstDimension
        : catFeaturesMatrixFirstDimension;

    if (documentCount == 0) {
        return 0;
    }

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is insufficient, must be at least document count * model prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    const size_t numericFeatureCount = hasNumericFeatures
        ? GetMatrixSecondDimension(jenv, jnumericFeaturesMatrix)
        : 0;
    const size_t catFeatureCount = hasCatFeatures
        ? GetMatrixSecondDimension(jenv, jcatFeaturesMatrix)
        : 0;

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;
    TVector<jintArray> catFeatureMatrixRowObjects;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    if (hasNumericFeatures) {
        numericFeatureMatrixRowObjects.reserve(documentCount);
        numericFeatureMatrixRows.reserve(documentCount);
    }

    if (hasCatFeatures) {
        catFeatureMatrixRowObjects.reserve(documentCount);
        catFeatureMatrixRows.reserve(documentCount);
    }

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                0);
        }
    };

    if (hasNumericFeatures) {
        for (size_t i = 0; i < documentCount; ++i) {
            numericFeatureMatrixRowObjects.push_back((jfloatArray)jenv->GetObjectArrayElement(
                jnumericFeaturesMatrix, i));
            const size_t rowSize = jenv->GetArrayLength(numericFeatureMatrixRowObjects.back());
            CB_ENSURE(
                numericFeatureCount == rowSize,
                "numeric feature count doesn't match for row " << i << ": "
                LabeledOutput(numericFeatureCount, rowSize));
            numericFeatureMatrixRows.push_back(MakeArrayRef(
                jenv->GetFloatArrayElements(numericFeatureMatrixRowObjects.back(), nullptr),
                numericFeatureCount));
        }
    }

    Y_SCOPE_EXIT(jenv, &catFeatureMatrixRowObjects, &catFeatureMatrixRows) {
        const auto size = catFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseIntArrayElements(
                catFeatureMatrixRowObjects[i],
                const_cast<jint*>(reinterpret_cast<const jint*>(catFeatureMatrixRows[i].data())),
                0);
        }
    };

    if (hasCatFeatures) {
        for (size_t i = 0; i < documentCount; ++i) {
            const auto row = (jobjectArray)jenv->GetObjectArrayElement(
                jcatFeaturesMatrix, i);
            const size_t rowSize = jenv->GetArrayLength(row);
            CB_ENSURE(
                catFeatureCount == rowSize,
                "cat feature count doesn't match for row " << i << ": "
                LabeledOutput(catFeatureCount, rowSize));
            catFeatureMatrixRows.push_back(MakeArrayRef(
                reinterpret_cast<const int*>(jenv->GetIntArrayElements(catFeatureMatrixRowObjects.back(), nullptr)),
                numericFeatureCount));
        }
    }

    TVector<double> predictions;
    predictions.yresize(documentCount * modelPredictionSize);
    model->Calc(numericFeatureMatrixRows, catFeatureMatrixRows, predictions);

    jenv->SetDoubleArrayRegion(jpredictions, 0, predictions.size(), predictions.data());

    Y_END_JNI_API_CALL();
}

#undef Y_BEGIN_JNI_API_CALL
#undef Y_END_JNI_API_CALL
