#include "ai_catboost_CatBoostJNIImpl.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>

#include <util/generic/cast.h>
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
    CB_ENSURE(catFeature, "OutOfMemoryError");
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

    Y_ASSERT(hashes.size() <= jcatFeaturesSize);

    for (size_t i = 0; i < hashes.size(); ++i) {
        // NOTE: instead of C-style cast `dynamic_cast` should be used, but compiler complains that
        // `_jobject` is not a polymorphic type
        const auto jcatFeature = (jstring)jenv->GetObjectArrayElement(jcatFeatures, i);
        CB_ENSURE(jenv->IsSameObject(jcatFeature, NULL) == JNI_FALSE, "got null array element");
        Y_SCOPE_EXIT(jenv, jcatFeature) {
            jenv->DeleteLocalRef(jcatFeature);
        };
        const auto catFeatureSize = jenv->GetStringUTFLength(jcatFeature);
        const auto* const catFeature = jenv->GetStringUTFChars(jcatFeature, nullptr);
        CB_ENSURE(catFeature, "OutOfMemoryError");
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
    CB_ENSURE(modelPath, "OutOfMemoryError");
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
    CB_ENSURE(data, "OutOfMemoryError");
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

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetTreeCount
  (JNIEnv* jenv, jclass, jlong jhandle, jintArray jtreeCount) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const jint treeCount = model->GetTreeCount();
    jenv->SetIntArrayRegion(jtreeCount, 0, 1, &treeCount);

    Y_END_JNI_API_CALL();
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelGetFeatureNames
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jfeatureNames) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    TVector<TString> featureNames = GetModelUsedFeaturesNames(*model);
    const size_t size = featureNames.size();

    for (size_t i = 0; i < size; ++i) {
        jstring jname = jenv->NewStringUTF(featureNames[i].c_str());
        jenv->SetObjectArrayElement(jfeatureNames, i, jname);
    }

    Y_END_JNI_API_CALL();
}

static size_t GetArraySize(JNIEnv* const jenv, const jarray array) {
    if (jenv->IsSameObject(array, NULL) == JNI_TRUE) {
        return 0;
    }

    return jenv->GetArrayLength(array);
}

JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3F_3Ljava_lang_String_2_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jfloatArray jnumericFeatures, jobjectArray jcatFeatures, jdoubleArray jprediction) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");
    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t numericFeatureCount = GetArraySize(jenv, jnumericFeatures);
    const size_t catFeatureCount = GetArraySize(jenv, jcatFeatures);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

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
                0);
        }
    };

    TVector<int> catFeatures;
    if (catFeatureCount) {
        catFeatures.yresize(catFeatureCount);
        HashCatFeatures(jenv, jcatFeatures, catFeatureCount, catFeatures);
    }

    TVector<double> prediction;
    prediction.yresize(modelPredictionSize);
    model->Calc(numericFeatures, catFeatures, prediction);

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

static size_t GetDocumentCount(JNIEnv* const jenv, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix ) {
    if (jenv->IsSameObject(jnumericFeaturesMatrix, NULL) != JNI_TRUE) {
        size_t documentCount = SafeIntegerCast<size_t>(jenv->GetArrayLength(jnumericFeaturesMatrix));
        if (jenv->IsSameObject(jcatFeaturesMatrix, NULL) != JNI_TRUE) {
            CB_ENSURE(
                SafeIntegerCast<size_t>(jenv->GetArrayLength(jcatFeaturesMatrix)) == documentCount,
                "numeric and cat features arrays have different number of objects");
        }
        return documentCount;
    } else if (jenv->IsSameObject(jcatFeaturesMatrix, NULL) != JNI_TRUE) {
        return SafeIntegerCast<size_t>(jenv->GetArrayLength(jcatFeaturesMatrix));
    } else {
        return 0;
    }
}


JNIEXPORT jstring JNICALL Java_ai_catboost_CatBoostJNIImpl_catBoostModelPredict__J_3_3F_3_3Ljava_lang_String_2_3D
  (JNIEnv* jenv, jclass, jlong jhandle, jobjectArray jnumericFeaturesMatrix, jobjectArray jcatFeaturesMatrix, jdoubleArray jpredictions) {
    Y_BEGIN_JNI_API_CALL();

    const auto* const model = ToConstFullModelPtr(jhandle);
    CB_ENSURE(model, "got nullptr model pointer");

    const size_t documentCount = GetDocumentCount(jenv, jnumericFeaturesMatrix, jcatFeaturesMatrix);
    if (documentCount == 0) {
        return 0;
    }

    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t numericFeatureCount = GetMatrixColumnCount(jenv, jnumericFeaturesMatrix);
    const size_t catFeatureCount = GetMatrixColumnCount(jenv, jcatFeaturesMatrix);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    if (numericFeatureCount && catFeatureCount) {
        const auto numericRows = jenv->GetArrayLength(jnumericFeaturesMatrix);
        const auto catRows = jenv->GetArrayLength(jcatFeaturesMatrix);
        CB_ENSURE(numericRows == catRows, LabeledOutput(numericRows, catRows));
    }

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is not sufficient, must be at least document count * prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;
    TVector<int> catFeatureMatrixRowwise;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                0);
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
    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t numericFeatureCount = GetArraySize(jenv, jnumericFeatures);
    const size_t catFeatureCount = GetArraySize(jenv, jcatFeatures);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

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
                0);
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

    const size_t documentCount = GetDocumentCount(jenv, jnumericFeaturesMatrix, jcatFeaturesMatrix);
    if (documentCount == 0) {
        return nullptr;
    }

    const size_t modelPredictionSize = model->GetDimensionsCount();
    const size_t minNumericFeatureCount = model->GetNumFloatFeatures();
    const size_t minCatFeatureCount = model->GetNumCatFeatures();
    const size_t numericFeatureCount = GetMatrixColumnCount(jenv, jnumericFeaturesMatrix);
    const size_t catFeatureCount = GetMatrixColumnCount(jenv, jcatFeaturesMatrix);

    CB_ENSURE(
        numericFeatureCount >= minNumericFeatureCount,
        LabeledOutput(numericFeatureCount, minNumericFeatureCount));

    CB_ENSURE(
        catFeatureCount >= minCatFeatureCount,
        LabeledOutput(catFeatureCount, minCatFeatureCount));

    if (numericFeatureCount && catFeatureCount) {
        const auto numericRows = jenv->GetArrayLength(jnumericFeaturesMatrix);
        const auto catRows = jenv->GetArrayLength(jcatFeaturesMatrix);
        CB_ENSURE(numericRows == catRows, LabeledOutput(numericRows, catRows));
    }

    const size_t predictionsSize = jenv->GetArrayLength(jpredictions);
    CB_ENSURE(
        predictionsSize >= documentCount * modelPredictionSize,
        "`prediction` size is insufficient, must be at least document count * model prediction dimension: "
        LabeledOutput(predictionsSize, documentCount * modelPredictionSize));

    TVector<jfloatArray> numericFeatureMatrixRowObjects;
    TVector<TConstArrayRef<float>> numericFeatureMatrixRows;
    TVector<jintArray> catFeatureMatrixRowObjects;
    TVector<TConstArrayRef<int>> catFeatureMatrixRows;

    Y_SCOPE_EXIT(jenv, &numericFeatureMatrixRowObjects, &numericFeatureMatrixRows) {
        const auto size = numericFeatureMatrixRows.size();
        for (size_t i = 0; i < size; ++i) {
            jenv->ReleaseFloatArrayElements(
                numericFeatureMatrixRowObjects[i],
                const_cast<float*>(numericFeatureMatrixRows[i].data()),
                0);
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
                0);
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

    TVector<double> predictions;
    predictions.yresize(documentCount * modelPredictionSize);
    model->Calc(numericFeatureMatrixRows, catFeatureMatrixRows, predictions);

    jenv->SetDoubleArrayRegion(jpredictions, 0, predictions.size(), predictions.data());

    Y_END_JNI_API_CALL();
}

#undef Y_BEGIN_JNI_API_CALL
#undef Y_END_JNI_API_CALL
