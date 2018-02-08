MAKEFLAGS += --no-builtin-rules
.ONESHELL:
PYTHON = $(dir $(shell which python))
BUILD_ROOT = $(shell pwd)
SOURCE_ROOT = $(shell pwd)

define _CC_TEST
__clang_major__ __clang_minor__
endef

_CC_VERSION = $(shell echo '$(_CC_TEST)' | $(CC) -E -P -)
$(info _CC_VERSION = '$(_CC_VERSION)')

ifneq '$(_CC_VERSION)' '3 9'
    $(error clang 3.9 is required)
endif

define _CXX_TEST
__clang_major__ __clang_minor__
endef

_CXX_VERSION = $(shell echo '$(_CXX_TEST)' | $(CXX) -E -P -)
$(info _CXX_VERSION = '$(_CXX_VERSION)')

ifneq '$(_CXX_VERSION)' '3 9'
    $(error clang 3.9 is required)
endif

all\
        ::\
        $(BUILD_ROOT)/catboost/app/catboost\
        $(BUILD_ROOT)/catboost/app/catboost.mf\


$(BUILD_ROOT)/catboost/app/catboost\
$(BUILD_ROOT)/catboost/app/catboost.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/util/charset/libutil-charset.a\
        $(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a\
        $(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/util/libyutil.a\
        $(BUILD_ROOT)/catboost/libs/cat_feature/libcatboost-libs-cat_feature.a\
        $(BUILD_ROOT)/library/colorizer/liblibrary-colorizer.a\
        $(BUILD_ROOT)/library/getopt/small/liblibrary-getopt-small.a\
        $(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
        $(BUILD_ROOT)/library/logger/liblibrary-logger.a\
        $(BUILD_ROOT)/library/logger/global/liblibrary-logger-global.a\
        $(BUILD_ROOT)/catboost/libs/logging/libcatboost-libs-logging.a\
        $(BUILD_ROOT)/contrib/libs/rapidjson/libcontrib-libs-rapidjson.a\
        $(BUILD_ROOT)/library/json/common/liblibrary-json-common.a\
        $(BUILD_ROOT)/library/json/writer/liblibrary-json-writer.a\
        $(BUILD_ROOT)/library/json/fast_sax/liblibrary-json-fast_sax.a\
        $(BUILD_ROOT)/library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a\
        $(BUILD_ROOT)/library/json/liblibrary-json.a\
        $(BUILD_ROOT)/catboost/libs/ctr_description/libcatboost-libs-ctr_description.a\
        $(BUILD_ROOT)/library/grid_creator/liblibrary-grid_creator.a\
        $(BUILD_ROOT)/catboost/libs/options/libcatboost-libs-options.a\
        $(BUILD_ROOT)/library/containers/2d_array/liblibrary-containers-2d_array.a\
        $(BUILD_ROOT)/library/binsaver/liblibrary-binsaver.a\
        $(BUILD_ROOT)/contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a\
        $(BUILD_ROOT)/contrib/libs/base64/avx2/liblibs-base64-avx2.a\
        $(BUILD_ROOT)/contrib/libs/base64/ssse3/liblibs-base64-ssse3.a\
        $(BUILD_ROOT)/contrib/libs/base64/neon32/liblibs-base64-neon32.a\
        $(BUILD_ROOT)/contrib/libs/base64/neon64/liblibs-base64-neon64.a\
        $(BUILD_ROOT)/contrib/libs/base64/plain32/liblibs-base64-plain32.a\
        $(BUILD_ROOT)/contrib/libs/base64/plain64/liblibs-base64-plain64.a\
        $(BUILD_ROOT)/library/string_utils/base64/liblibrary-string_utils-base64.a\
        $(BUILD_ROOT)/library/digest/md5/liblibrary-digest-md5.a\
        $(BUILD_ROOT)/contrib/libs/crcutil/libcontrib-libs-crcutil.a\
        $(BUILD_ROOT)/library/digest/crc32c/liblibrary-digest-crc32c.a\
        $(BUILD_ROOT)/library/threading/local_executor/liblibrary-threading-local_executor.a\
        $(BUILD_ROOT)/catboost/libs/helpers/libcatboost-libs-helpers.a\
        $(BUILD_ROOT)/catboost/libs/column_description/libcatboost-libs-column_description.a\
        $(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a\
        $(BUILD_ROOT)/contrib/libs/coreml/libcontrib-libs-coreml.a\
        $(BUILD_ROOT)/library/containers/dense_hash/liblibrary-containers-dense_hash.a\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a\
        $(BUILD_ROOT)/catboost/libs/model/libcatboost-libs-model.a\
        $(BUILD_ROOT)/catboost/libs/data/libcatboost-libs-data.a\
        $(BUILD_ROOT)/contrib/libs/tensorboard/libcontrib-libs-tensorboard.a\
        $(BUILD_ROOT)/catboost/libs/loggers/libcatboost-libs-loggers.a\
        $(BUILD_ROOT)/catboost/libs/metrics/libcatboost-libs-metrics.a\
        $(BUILD_ROOT)/library/statistics/liblibrary-statistics.a\
        $(BUILD_ROOT)/catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a\
        $(BUILD_ROOT)/library/dot_product/liblibrary-dot_product.a\
        $(BUILD_ROOT)/contrib/libs/fmath/libcontrib-libs-fmath.a\
        $(BUILD_ROOT)/library/fast_exp/liblibrary-fast_exp.a\
        $(BUILD_ROOT)/library/fast_log/liblibrary-fast_log.a\
        $(BUILD_ROOT)/library/object_factory/liblibrary-object_factory.a\
        $(BUILD_ROOT)/catboost/libs/algo/libcatboost-libs-algo.a\
        $(BUILD_ROOT)/catboost/libs/fstr/libcatboost-libs-fstr.a\
        $(BUILD_ROOT)/catboost/libs/train_lib/train_model.cpp.o\
        $(BUILD_ROOT)/catboost/libs/train_lib/libcatboost-libs-train_lib.a\
        $(BUILD_ROOT)/library/svnversion/liblibrary-svnversion.a\
        $(BUILD_ROOT)/catboost/app/cmd_line.cpp.o\
        $(BUILD_ROOT)/catboost/app/bind_options.cpp.o\
        $(BUILD_ROOT)/catboost/app/mode_plot.cpp.o\
        $(BUILD_ROOT)/catboost/app/mode_fstr.cpp.o\
        $(BUILD_ROOT)/catboost/app/mode_fit.cpp.o\
        $(BUILD_ROOT)/catboost/app/mode_calc.cpp.o\
        $(BUILD_ROOT)/catboost/app/main.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost -o catboost/app/catboost.mf -t PROGRAM --no-gpl -Ya,lics -Ya,peers catboost/libs/algo/libcatboost-libs-algo.a catboost/libs/train_lib/libcatboost-libs-train_lib.a catboost/libs/data/libcatboost-libs-data.a catboost/libs/fstr/libcatboost-libs-fstr.a catboost/libs/helpers/libcatboost-libs-helpers.a catboost/libs/logging/libcatboost-libs-logging.a catboost/libs/model/libcatboost-libs-model.a catboost/libs/options/libcatboost-libs-options.a library/getopt/small/liblibrary-getopt-small.a library/grid_creator/liblibrary-grid_creator.a library/json/liblibrary-json.a library/svnversion/liblibrary-svnversion.a library/threading/local_executor/liblibrary-threading-local_executor.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a catboost/libs/loggers/libcatboost-libs-loggers.a catboost/libs/metrics/libcatboost-libs-metrics.a catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a library/binsaver/liblibrary-binsaver.a library/containers/2d_array/liblibrary-containers-2d_array.a library/containers/dense_hash/liblibrary-containers-dense_hash.a library/digest/md5/liblibrary-digest-md5.a library/dot_product/liblibrary-dot_product.a library/fast_exp/liblibrary-fast_exp.a library/fast_log/liblibrary-fast_log.a library/object_factory/liblibrary-object_factory.a catboost/libs/cat_feature/libcatboost-libs-cat_feature.a catboost/libs/column_description/libcatboost-libs-column_description.a library/digest/crc32c/liblibrary-digest-crc32c.a library/malloc/api/liblibrary-malloc-api.a library/logger/liblibrary-logger.a library/logger/global/liblibrary-logger-global.a catboost/libs/ctr_description/libcatboost-libs-ctr_description.a contrib/libs/coreml/libcontrib-libs-coreml.a catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a library/colorizer/liblibrary-colorizer.a contrib/libs/rapidjson/libcontrib-libs-rapidjson.a library/json/writer/liblibrary-json-writer.a library/json/common/liblibrary-json-common.a library/json/fast_sax/liblibrary-json-fast_sax.a library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a contrib/libs/tensorboard/libcontrib-libs-tensorboard.a library/statistics/liblibrary-statistics.a contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a library/string_utils/base64/liblibrary-string_utils-base64.a contrib/libs/fmath/libcontrib-libs-fmath.a contrib/libs/crcutil/libcontrib-libs-crcutil.a contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/base64/avx2/liblibs-base64-avx2.a contrib/libs/base64/ssse3/liblibs-base64-ssse3.a contrib/libs/base64/neon32/liblibs-base64-neon32.a contrib/libs/base64/neon64/liblibs-base64-neon64.a contrib/libs/base64/plain32/liblibs-base64-plain32.a contrib/libs/base64/plain64/liblibs-base64-plain64.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/catboost/app/cmd_line.cpp.o' '$(BUILD_ROOT)/catboost/app/bind_options.cpp.o' '$(BUILD_ROOT)/catboost/app/mode_plot.cpp.o' '$(BUILD_ROOT)/catboost/app/mode_fstr.cpp.o' '$(BUILD_ROOT)/catboost/app/mode_fit.cpp.o' '$(BUILD_ROOT)/catboost/app/mode_calc.cpp.o' '$(BUILD_ROOT)/catboost/app/main.cpp.o' -o '$(BUILD_ROOT)/catboost/app/catboost' -rdynamic catboost/libs/train_lib/train_model.cpp.o -Wl,--start-group catboost/libs/algo/libcatboost-libs-algo.a catboost/libs/train_lib/libcatboost-libs-train_lib.a catboost/libs/data/libcatboost-libs-data.a catboost/libs/fstr/libcatboost-libs-fstr.a catboost/libs/helpers/libcatboost-libs-helpers.a catboost/libs/logging/libcatboost-libs-logging.a catboost/libs/model/libcatboost-libs-model.a catboost/libs/options/libcatboost-libs-options.a library/getopt/small/liblibrary-getopt-small.a library/grid_creator/liblibrary-grid_creator.a library/json/liblibrary-json.a library/svnversion/liblibrary-svnversion.a library/threading/local_executor/liblibrary-threading-local_executor.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a catboost/libs/loggers/libcatboost-libs-loggers.a catboost/libs/metrics/libcatboost-libs-metrics.a catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a library/binsaver/liblibrary-binsaver.a library/containers/2d_array/liblibrary-containers-2d_array.a library/containers/dense_hash/liblibrary-containers-dense_hash.a library/digest/md5/liblibrary-digest-md5.a library/dot_product/liblibrary-dot_product.a library/fast_exp/liblibrary-fast_exp.a library/fast_log/liblibrary-fast_log.a library/object_factory/liblibrary-object_factory.a catboost/libs/cat_feature/libcatboost-libs-cat_feature.a catboost/libs/column_description/libcatboost-libs-column_description.a library/digest/crc32c/liblibrary-digest-crc32c.a library/malloc/api/liblibrary-malloc-api.a library/logger/liblibrary-logger.a library/logger/global/liblibrary-logger-global.a catboost/libs/ctr_description/libcatboost-libs-ctr_description.a contrib/libs/coreml/libcontrib-libs-coreml.a catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a library/colorizer/liblibrary-colorizer.a contrib/libs/rapidjson/libcontrib-libs-rapidjson.a library/json/writer/liblibrary-json-writer.a library/json/common/liblibrary-json-common.a library/json/fast_sax/liblibrary-json-fast_sax.a library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a contrib/libs/tensorboard/libcontrib-libs-tensorboard.a library/statistics/liblibrary-statistics.a contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a library/string_utils/base64/liblibrary-string_utils-base64.a contrib/libs/fmath/libcontrib-libs-fmath.a contrib/libs/crcutil/libcontrib-libs-crcutil.a contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/base64/avx2/liblibs-base64-avx2.a contrib/libs/base64/ssse3/liblibs-base64-ssse3.a contrib/libs/base64/neon32/liblibs-base64-neon32.a contrib/libs/base64/neon64/liblibs-base64-neon64.a contrib/libs/base64/plain32/liblibs-base64-plain32.a contrib/libs/base64/plain64/liblibs-base64-plain64.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lrt -ldl -lpthread -nodefaultlibs -lpthread -lc -lm

$(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
$(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/demangle.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cppdemangle'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-cppdemangle -o contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a.mf -t LIBRARY -Ya,lics MIT BSD -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a' '$(BUILD_ROOT)/contrib/libs/cppdemangle/demangle.cpp.o'

$(BUILD_ROOT)/contrib/libs/cppdemangle/demangle.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cppdemangle/demangle.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cppdemangle'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cppdemangle/demangle.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cppdemangle/demangle.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
$(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp.o\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-libunwind_master -o contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp.o' '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp.o'

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersSave.S' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include'

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindRegistersRestore.S' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include'

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/Unwind-sjlj.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -std=c99

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1-gcc-ext.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -std=c99

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/UnwindLevel1.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -std=c99

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/Unwind-EHABI.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -fno-rtti -fno-exceptions -funwind-tables -nostdinc++

$(BUILD_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/libunwind_master/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp.o' '$(SOURCE_ROOT)/contrib/libs/libunwind_master/src/libunwind.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/libunwind_master/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -fno-rtti -fno-exceptions -funwind-tables -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-cxxsupp-builtins -o contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a.mf -t LIBRARY -Ya,lics MIT BSD -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c.o'

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/gcc_personality_v0.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/int_util.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/floatuntidf.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/fixunsdfti.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/clzti2.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divdc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divxc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/divsc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/mulxc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/muldc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/mulsc3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/umodti3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/udivti3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/builtins/udivmodti4.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-cxxsupp-libcxxrt -o contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a.mf -t LIBRARY -Ya,lics BSD -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc.o'

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/unwind.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/dynamic_cast.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/typeinfo.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/guard.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/exception.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/stdexcept.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/auxhelper.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt/memory.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++ -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp.o\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-cxxsupp-libcxx -o contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a.mf -t LIBRARY -Ya,lics MIT BSD -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp.o' '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp.o'

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/new.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/vector.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/variant.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/valarray.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/utility.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/typeinfo.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/thread.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/system_error.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/strstream.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/string.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/stdexcept.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/shared_mutex.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/regex.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/random.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/optional.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/mutex.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/memory.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/locale.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/iostream.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/ios.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/hash.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/future.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/functional.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/exception.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/debug.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/condition_variable.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/chrono.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/bind.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/any.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp.o' '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/src/algorithm.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -D_LIBCPP_BUILDING_LIBRARY -DLIBCXXRT=1 -nostdinc++

$(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
$(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/__/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-cxxsupp -o contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a' '$(BUILD_ROOT)/contrib/libs/cxxsupp/__/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/contrib/libs/cxxsupp/__/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/cxxsupp/__/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/cxxsupp/__/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/charset/libutil-charset.a\
$(BUILD_ROOT)/util/charset/libutil-charset.a.mf\
        ::\
        $(BUILD_ROOT)/util/charset/wide_sse41.cpp.o\
        $(BUILD_ROOT)/util/charset/all_charset.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/util/charset'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name util-charset -o util/charset/libutil-charset.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/util/charset/libutil-charset.a' '$(BUILD_ROOT)/util/charset/wide_sse41.cpp.o' '$(BUILD_ROOT)/util/charset/all_charset.cpp.o'

$(BUILD_ROOT)/util/charset/wide_sse41.cpp.o\
        ::\
        $(SOURCE_ROOT)/util/charset/wide_sse41.cpp\

	mkdir -p '$(BUILD_ROOT)/util/charset'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/charset/wide_sse41.cpp.o' '$(SOURCE_ROOT)/util/charset/wide_sse41.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++ -msse4.1

$(BUILD_ROOT)/util/charset/all_charset.cpp.o\
        ::\
        $(BUILD_ROOT)/util/charset/all_charset.cpp\

	mkdir -p '$(BUILD_ROOT)/util/charset'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/charset/all_charset.cpp.o' '$(BUILD_ROOT)/util/charset/all_charset.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/charset/all_charset.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/charset/generated/unidata.cpp\
        $(SOURCE_ROOT)/util/charset/recode_result.cpp\
        $(SOURCE_ROOT)/util/charset/unicode_table.cpp\
        $(SOURCE_ROOT)/util/charset/unidata.cpp\
        $(SOURCE_ROOT)/util/charset/utf8.cpp\
        $(SOURCE_ROOT)/util/charset/wide.cpp\

	mkdir -p '$(BUILD_ROOT)/util/charset'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/charset/all_charset.cpp' util/charset/generated/unidata.cpp util/charset/recode_result.cpp util/charset/unicode_table.cpp util/charset/unidata.cpp util/charset/utf8.cpp util/charset/wide.cpp

$(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a\
$(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/zlib/zutil.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/uncompr.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/trees.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/inftrees.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/inflate.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/inffast.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/infback.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/gzwrite.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/gzread.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/gzlib.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/gzclose.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/deflate.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/crc32.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/compress.c.o\
        $(BUILD_ROOT)/contrib/libs/zlib/adler32.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-zlib -o contrib/libs/zlib/libcontrib-libs-zlib.a.mf -t LIBRARY -Ya,lics ZLIB -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a' '$(BUILD_ROOT)/contrib/libs/zlib/zutil.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/uncompr.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/trees.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/inftrees.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/inflate.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/inffast.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/infback.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/gzwrite.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/gzread.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/gzlib.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/gzclose.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/deflate.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/crc32.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/compress.c.o' '$(BUILD_ROOT)/contrib/libs/zlib/adler32.c.o'

$(BUILD_ROOT)/contrib/libs/zlib/zutil.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/zutil.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/zutil.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/zutil.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/uncompr.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/uncompr.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/uncompr.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/uncompr.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/trees.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/trees.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/trees.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/trees.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/inftrees.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/inftrees.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/inftrees.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/inftrees.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/inflate.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/inflate.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/inflate.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/inflate.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/inffast.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/inffast.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/inffast.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/inffast.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/infback.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/infback.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/infback.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/infback.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/gzwrite.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/gzwrite.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/gzwrite.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/gzwrite.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/gzread.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/gzread.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/gzread.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/gzread.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/gzlib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/gzlib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/gzlib.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/gzlib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/gzclose.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/gzclose.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/gzclose.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/gzclose.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/deflate.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/deflate.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/deflate.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/deflate.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/crc32.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/crc32.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/crc32.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/crc32.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/compress.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/compress.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/compress.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/compress.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/zlib/adler32.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/zlib/adler32.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/zlib'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/zlib/adler32.c.o' '$(SOURCE_ROOT)/contrib/libs/zlib/adler32.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a\
$(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/bignum.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/strtod.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/diy-fp.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/double-conversion.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc.o\
        $(BUILD_ROOT)/contrib/libs/double-conversion/cached-powers.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-double-conversion -o contrib/libs/double-conversion/libcontrib-libs-double-conversion.a.mf -t LIBRARY -Ya,lics BSD3 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a' '$(BUILD_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/bignum.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/strtod.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/diy-fp.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/double-conversion.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc.o' '$(BUILD_ROOT)/contrib/libs/double-conversion/cached-powers.cc.o'

$(BUILD_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/fast-dtoa.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/bignum.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/bignum.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/bignum.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/bignum.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/strtod.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/strtod.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/strtod.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/strtod.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/fixed-dtoa.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/diy-fp.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/diy-fp.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/diy-fp.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/diy-fp.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/double-conversion.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/double-conversion.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/double-conversion.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/double-conversion.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/bignum-dtoa.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/double-conversion/cached-powers.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/double-conversion/cached-powers.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/double-conversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/double-conversion/cached-powers.cc.o' '$(SOURCE_ROOT)/contrib/libs/double-conversion/cached-powers.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
$(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a.mf\
        ::\
        $(BUILD_ROOT)/library/malloc/api/malloc.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/malloc/api'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-malloc-api -o library/malloc/api/liblibrary-malloc-api.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a' '$(BUILD_ROOT)/library/malloc/api/malloc.cpp.o'

$(BUILD_ROOT)/library/malloc/api/malloc.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/malloc/api/malloc.cpp\

	mkdir -p '$(BUILD_ROOT)/library/malloc/api'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/malloc/api/malloc.cpp.o' '$(SOURCE_ROOT)/library/malloc/api/malloc.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/libyutil.a\
$(BUILD_ROOT)/util/libyutil.a.mf\
        ::\
        $(BUILD_ROOT)/util/system/mktemp_system.cpp.o\
        $(BUILD_ROOT)/util/system/strlcpy.c.o\
        $(BUILD_ROOT)/util/system/valgrind.cpp.o\
        $(BUILD_ROOT)/util/system/context_x86.o\
        $(BUILD_ROOT)/util/string/cast.cc.o\
        $(BUILD_ROOT)/util/random/random.cpp.o\
        $(BUILD_ROOT)/util/digest/city.cpp.o\
        $(BUILD_ROOT)/util/datetime/parser.rl6.cpp.o\
        $(BUILD_ROOT)/util/all_thread.cpp.o\
        $(BUILD_ROOT)/util/all_system_2.cpp.o\
        $(BUILD_ROOT)/util/all_system_1.cpp.o\
        $(BUILD_ROOT)/util/all_string.cpp.o\
        $(BUILD_ROOT)/util/all_stream.cpp.o\
        $(BUILD_ROOT)/util/all_random.cpp.o\
        $(BUILD_ROOT)/util/all_network.cpp.o\
        $(BUILD_ROOT)/util/all_memory.cpp.o\
        $(BUILD_ROOT)/util/all_generic.cpp.o\
        $(BUILD_ROOT)/util/all_folder.cpp.o\
        $(BUILD_ROOT)/util/all_util.cpp.o\
        $(BUILD_ROOT)/util/all_digest.cpp.o\
        $(BUILD_ROOT)/util/all_datetime.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name yutil -o util/libyutil.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/util/libyutil.a' '$(BUILD_ROOT)/util/system/mktemp_system.cpp.o' '$(BUILD_ROOT)/util/system/strlcpy.c.o' '$(BUILD_ROOT)/util/system/valgrind.cpp.o' '$(BUILD_ROOT)/util/system/context_x86.o' '$(BUILD_ROOT)/util/string/cast.cc.o' '$(BUILD_ROOT)/util/random/random.cpp.o' '$(BUILD_ROOT)/util/digest/city.cpp.o' '$(BUILD_ROOT)/util/datetime/parser.rl6.cpp.o' '$(BUILD_ROOT)/util/all_thread.cpp.o' '$(BUILD_ROOT)/util/all_system_2.cpp.o' '$(BUILD_ROOT)/util/all_system_1.cpp.o' '$(BUILD_ROOT)/util/all_string.cpp.o' '$(BUILD_ROOT)/util/all_stream.cpp.o' '$(BUILD_ROOT)/util/all_random.cpp.o' '$(BUILD_ROOT)/util/all_network.cpp.o' '$(BUILD_ROOT)/util/all_memory.cpp.o' '$(BUILD_ROOT)/util/all_generic.cpp.o' '$(BUILD_ROOT)/util/all_folder.cpp.o' '$(BUILD_ROOT)/util/all_util.cpp.o' '$(BUILD_ROOT)/util/all_digest.cpp.o' '$(BUILD_ROOT)/util/all_datetime.cpp.o'

$(BUILD_ROOT)/util/system/mktemp_system.cpp.o\
        ::\
        $(SOURCE_ROOT)/util/system/mktemp_system.cpp\

	mkdir -p '$(BUILD_ROOT)/util/system'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/system/mktemp_system.cpp.o' '$(SOURCE_ROOT)/util/system/mktemp_system.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/system/strlcpy.c.o\
        ::\
        $(SOURCE_ROOT)/util/system/strlcpy.c\

	mkdir -p '$(BUILD_ROOT)/util/system'
	'$(CC)' -c -o '$(BUILD_ROOT)/util/system/strlcpy.c.o' '$(SOURCE_ROOT)/util/system/strlcpy.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)'

$(BUILD_ROOT)/util/system/valgrind.cpp.o\
        ::\
        $(SOURCE_ROOT)/util/system/valgrind.cpp\

	mkdir -p '$(BUILD_ROOT)/util/system'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/system/valgrind.cpp.o' '$(SOURCE_ROOT)/util/system/valgrind.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/system/context_x86.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/yasm/yasm\
        $(SOURCE_ROOT)/util/system/context_x86.asm\

	mkdir -p '$(BUILD_ROOT)/util/system'
	'$(BUILD_ROOT)/contrib/tools/yasm/yasm' -f elf64 -D UNIX -D _x86_64_ -D_YASM_ -g dwarf2 -I '$(BUILD_ROOT)' -I '$(SOURCE_ROOT)' -I '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' -I '$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -o '$(BUILD_ROOT)/util/system/context_x86.o' '$(SOURCE_ROOT)/util/system/context_x86.asm'

$(BUILD_ROOT)/contrib/tools/yasm/yasm\
$(BUILD_ROOT)/contrib/tools/yasm/yasm.mf\
        ::\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/x86regtmod.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/x86cpu.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/nasm-token.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/lc3bid.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/init_plugin.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/gas-token.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/value.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/valparam.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/symrec.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/strsep.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/section.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/phash.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/mergesort.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/md5.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/linemap.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/inttree.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/intnum.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/insn.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/hamt.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/floatnum.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/file.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/expr.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/errwarn.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bytecode.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bitvect.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-org.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-data.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-align.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/libyasm/assocdat.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c.o\
        $(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name yasm -o contrib/tools/yasm/yasm.mf -t PROGRAM -Ya,lics -Ya,peers
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/contrib/tools/yasm/modules/x86regtmod.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/x86cpu.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/nasm-token.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/lc3bid.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/init_plugin.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/gas-token.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/value.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/valparam.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/symrec.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strsep.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/section.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/phash.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/mergesort.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/md5.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/linemap.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/inttree.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/intnum.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/insn.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/hamt.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/floatnum.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/file.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/expr.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/errwarn.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bytecode.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bitvect.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-org.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-data.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-align.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/assocdat.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c.o' '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c.o' -o '$(BUILD_ROOT)/contrib/tools/yasm/yasm' -rdynamic -Wl,--start-group -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -nodefaultlibs -lpthread -lc -lm -s

$(BUILD_ROOT)/contrib/tools/yasm/modules/x86regtmod.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/x86regtmod.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/x86regtmod.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/x86regtmod.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/x86cpu.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/x86cpu.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/x86cpu.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/x86cpu.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/raw'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/raw/raw-preproc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasmlib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-preproc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-pp.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/nasm/nasm-eval.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-preproc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/gas/gas-eval.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/cpp'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/preprocs/cpp/cpp-preproc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parser.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/nasm/nasm-parse.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parser.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/parsers/gas/gas-parse-intel.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/xdf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/xdf/xdf-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/rdf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/rdf/rdf-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/macho'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/macho/macho-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x86.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-x32.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-x86-amd64.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/elf/elf-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/dbg'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/dbg/dbg-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/coff/win64-except.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/coff/coff-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/bin'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/objfmts/bin/bin-objfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/nasm-token.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/nasm-token.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/nasm-token.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/nasm-token.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/listfmts/nasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/listfmts/nasm/nasm-listfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/lc3bid.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/lc3bid.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/lc3bid.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/lc3bid.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/init_plugin.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/init_plugin.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/init_plugin.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/init_plugin.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/gas-token.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/gas-token.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/gas-token.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/gas-token.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/stabs/stabs-dbgfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/null'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/null/null-dbgfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-line.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-info.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/dwarf2/dwarf2-aranges.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-type.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-symline.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/dbgfmts/codeview/cv-dbgfmt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86id.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86expr.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86bc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/x86/x86arch.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3bbc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/modules/arch/lc3b/lc3barch.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/xstrdup.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/xmalloc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/value.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/value.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/value.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/value.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/valparam.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/valparam.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/valparam.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/valparam.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/symrec.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/symrec.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/symrec.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/symrec.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strsep.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/strsep.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strsep.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/strsep.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/strcasecmp.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/section.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/section.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/section.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/section.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/phash.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/phash.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/phash.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/phash.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/mergesort.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/mergesort.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/mergesort.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/mergesort.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/md5.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/md5.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/md5.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/md5.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/linemap.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/linemap.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/linemap.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/linemap.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/inttree.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/inttree.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/inttree.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/inttree.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/intnum.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/intnum.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/intnum.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/intnum.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/insn.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/insn.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/insn.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/insn.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/hamt.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/hamt.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/hamt.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/hamt.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/floatnum.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/floatnum.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/floatnum.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/floatnum.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/file.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/file.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/file.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/file.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/expr.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/expr.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/expr.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/expr.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/errwarn.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/errwarn.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/errwarn.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/errwarn.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/cmake-module.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bytecode.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bytecode.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bytecode.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bytecode.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bitvect.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bitvect.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bitvect.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bitvect.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-reserve.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-org.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-org.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-org.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-org.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-incbin.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-data.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-data.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-data.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-data.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-align.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-align.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/bc-align.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/bc-align.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/libyasm/assocdat.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/libyasm/assocdat.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/libyasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/libyasm/assocdat.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/libyasm/assocdat.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm/yasm.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c.o' '$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm/yasm-options.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/frontends/yasm' '-I$(SOURCE_ROOT)/contrib/tools/yasm/modules' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_CONFIG_H -DYASM_LIB_SOURCE -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/util/string/cast.cc.o\
        ::\
        $(SOURCE_ROOT)/util/string/cast.cc\

	mkdir -p '$(BUILD_ROOT)/util/string'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/string/cast.cc.o' '$(SOURCE_ROOT)/util/string/cast.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/random/random.cpp.o\
        ::\
        $(SOURCE_ROOT)/util/random/random.cpp\

	mkdir -p '$(BUILD_ROOT)/util/random'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/random/random.cpp.o' '$(SOURCE_ROOT)/util/random/random.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/digest/city.cpp.o\
        ::\
        $(SOURCE_ROOT)/util/digest/city.cpp\

	mkdir -p '$(BUILD_ROOT)/util/digest'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/digest/city.cpp.o' '$(SOURCE_ROOT)/util/digest/city.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/datetime/parser.rl6.cpp.o\
        ::\
        $(BUILD_ROOT)/util/datetime/parser.rl6.cpp\

	mkdir -p '$(BUILD_ROOT)/util/datetime'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/datetime/parser.rl6.cpp.o' '$(BUILD_ROOT)/util/datetime/parser.rl6.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/datetime/parser.rl6.cpp\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/ragel6\
        $(SOURCE_ROOT)/util/datetime/parser.rl6\

	mkdir -p '$(BUILD_ROOT)/util/datetime'
	'$(BUILD_ROOT)/contrib/tools/ragel6/ragel6' -CT0 '-I$(SOURCE_ROOT)' -o '$(BUILD_ROOT)/util/datetime/parser.rl6.cpp' '$(SOURCE_ROOT)/util/datetime/parser.rl6'

$(BUILD_ROOT)/contrib/tools/ragel6/ragel6\
$(BUILD_ROOT)/contrib/tools/ragel6/ragel6.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/contrib/libs/jemalloc/libcontrib-libs-jemalloc.a\
        $(BUILD_ROOT)/library/malloc/jemalloc/liblibrary-malloc-jemalloc.a\
        $(BUILD_ROOT)/contrib/tools/ragel6/rlscan.cpp.o\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp.o\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp.o\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp.o\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp.o\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name ragel6 -o contrib/tools/ragel6/ragel6.mf -t PROGRAM -Ya,lics -Ya,peers contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a library/malloc/jemalloc/liblibrary-malloc-jemalloc.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/jemalloc/libcontrib-libs-jemalloc.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/contrib/tools/ragel6/rlscan.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp.o' -o '$(BUILD_ROOT)/contrib/tools/ragel6/ragel6' -rdynamic -Wl,--start-group contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a library/malloc/jemalloc/liblibrary-malloc-jemalloc.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/jemalloc/libcontrib-libs-jemalloc.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lpthread -nodefaultlibs -lpthread -lc -lm -s

$(BUILD_ROOT)/contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a\
$(BUILD_ROOT)/contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel5/aapl/__/__/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel5/aapl'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name tools-ragel5-aapl -o contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a.mf -t LIBRARY -Ya,lics LGPL -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/tools/ragel5/aapl/libtools-ragel5-aapl.a' '$(BUILD_ROOT)/contrib/tools/ragel5/aapl/__/__/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/contrib/tools/ragel5/aapl/__/__/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel5/aapl/__/__/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel5/aapl/__/__/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/jemalloc/libcontrib-libs-jemalloc.a\
$(BUILD_ROOT)/contrib/libs/jemalloc/libcontrib-libs-jemalloc.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/jemalloc/hack.cpp.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/util.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/tsd.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/tcache.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/stats.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/rtree.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/quarantine.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/prof.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/mutex.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/mb.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/jemalloc.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/huge.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/hash.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/extent.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/ctl.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/ckh.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/bitmap.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/base.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/atomic.c.o\
        $(BUILD_ROOT)/contrib/libs/jemalloc/src/arena.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-jemalloc -o contrib/libs/jemalloc/libcontrib-libs-jemalloc.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/jemalloc/libcontrib-libs-jemalloc.a' '$(BUILD_ROOT)/contrib/libs/jemalloc/hack.cpp.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/util.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/tsd.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/tcache.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/stats.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/rtree.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/quarantine.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/prof.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/mutex.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/mb.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/jemalloc.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/huge.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/hash.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/extent.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/ctl.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/ckh.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/bitmap.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/base.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/atomic.c.o' '$(BUILD_ROOT)/contrib/libs/jemalloc/src/arena.c.o'

$(BUILD_ROOT)/contrib/libs/jemalloc/hack.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/hack.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/hack.cpp.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/hack.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/jemalloc/src/util.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/util.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/util.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/util.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/tsd.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/tsd.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/tsd.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/tsd.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/tcache.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/tcache.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/tcache.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/tcache.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/stats.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/stats.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/stats.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/stats.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/rtree.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/rtree.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/rtree.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/rtree.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/quarantine.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/quarantine.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/quarantine.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/quarantine.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/prof.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/prof.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/prof.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/prof.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/mutex.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/mutex.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/mutex.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/mutex.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/mb.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/mb.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/mb.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/mb.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/jemalloc.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/jemalloc.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/jemalloc.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/jemalloc.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/huge.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/huge.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/huge.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/huge.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/hash.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/hash.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/hash.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/hash.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/extent.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/extent.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/extent.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/extent.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/ctl.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/ctl.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/ctl.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/ctl.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/ckh.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/ckh.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/ckh.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/ckh.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk_mmap.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk_dss.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/chunk.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/chunk.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/bitmap.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/bitmap.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/bitmap.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/bitmap.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/base.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/base.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/base.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/base.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/atomic.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/atomic.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/atomic.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/atomic.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/contrib/libs/jemalloc/src/arena.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/jemalloc/src/arena.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/jemalloc/src'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/jemalloc/src/arena.c.o' '$(SOURCE_ROOT)/contrib/libs/jemalloc/src/arena.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/jemalloc/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -fvisibility=hidden -funroll-loops -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow

$(BUILD_ROOT)/library/malloc/jemalloc/liblibrary-malloc-jemalloc.a\
$(BUILD_ROOT)/library/malloc/jemalloc/liblibrary-malloc-jemalloc.a.mf\
        ::\
        $(BUILD_ROOT)/library/malloc/jemalloc/malloc-info.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/malloc/jemalloc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-malloc-jemalloc -o library/malloc/jemalloc/liblibrary-malloc-jemalloc.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/malloc/jemalloc/liblibrary-malloc-jemalloc.a' '$(BUILD_ROOT)/library/malloc/jemalloc/malloc-info.cpp.o'

$(BUILD_ROOT)/library/malloc/jemalloc/malloc-info.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/malloc/jemalloc/malloc-info.cpp\

	mkdir -p '$(BUILD_ROOT)/library/malloc/jemalloc'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/malloc/jemalloc/malloc-info.cpp.o' '$(SOURCE_ROOT)/library/malloc/jemalloc/malloc-info.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/rlscan.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rlscan.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/rlscan.cpp.o' '$(SOURCE_ROOT)/contrib/tools/ragel6/rlscan.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/contrib/tools/ragel6/dotcodegen.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmgraph.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/inputdata.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/parsetree.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/redfsm.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/javacodegen.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rbxgoto.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src5.cpp' contrib/tools/ragel6/dotcodegen.cpp contrib/tools/ragel6/fsmgraph.cpp contrib/tools/ragel6/inputdata.cpp contrib/tools/ragel6/parsetree.cpp contrib/tools/ragel6/redfsm.cpp contrib/tools/ragel6/javacodegen.cpp contrib/tools/ragel6/rbxgoto.cpp

$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rubyfflat.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdfgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdipgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cscodegen.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csftable.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cstable.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmbase.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/gendata.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rubyftable.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src4.cpp' contrib/tools/ragel6/rubyfflat.cpp contrib/tools/ragel6/cdfgoto.cpp contrib/tools/ragel6/cdipgoto.cpp contrib/tools/ragel6/cscodegen.cpp contrib/tools/ragel6/csftable.cpp contrib/tools/ragel6/cstable.cpp contrib/tools/ragel6/fsmbase.cpp contrib/tools/ragel6/gendata.cpp contrib/tools/ragel6/rubyftable.cpp

$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdcodegen.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdftable.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdtable.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csfgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csipgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmap.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmmin.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmattach.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csfflat.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rubyflat.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src3.cpp' contrib/tools/ragel6/cdcodegen.cpp contrib/tools/ragel6/cdftable.cpp contrib/tools/ragel6/cdtable.cpp contrib/tools/ragel6/csfgoto.cpp contrib/tools/ragel6/csipgoto.cpp contrib/tools/ragel6/fsmap.cpp contrib/tools/ragel6/fsmmin.cpp contrib/tools/ragel6/fsmattach.cpp contrib/tools/ragel6/csfflat.cpp contrib/tools/ragel6/rubyflat.cpp

$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rlparse.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdflat.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdsplit.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/fsmstate.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/main.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/xmlcodegen.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src2.cpp' contrib/tools/ragel6/rlparse.cpp contrib/tools/ragel6/cdflat.cpp contrib/tools/ragel6/cdsplit.cpp contrib/tools/ragel6/csgoto.cpp contrib/tools/ragel6/fsmstate.cpp contrib/tools/ragel6/main.cpp contrib/tools/ragel6/xmlcodegen.cpp

$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp.o' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/tools/ragel5/aapl' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -O2 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rubycodegen.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/rubytable.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdfflat.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cdgoto.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/common.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/csflat.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/cssplit.cpp\
        $(SOURCE_ROOT)/contrib/tools/ragel6/parsedata.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/ragel6'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/contrib/tools/ragel6/all_src1.cpp' contrib/tools/ragel6/rubycodegen.cpp contrib/tools/ragel6/rubytable.cpp contrib/tools/ragel6/cdfflat.cpp contrib/tools/ragel6/cdgoto.cpp contrib/tools/ragel6/common.cpp contrib/tools/ragel6/csflat.cpp contrib/tools/ragel6/cssplit.cpp contrib/tools/ragel6/parsedata.cpp

$(BUILD_ROOT)/util/all_thread.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_thread.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_thread.cpp.o' '$(BUILD_ROOT)/util/all_thread.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_thread.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/thread/pool.cpp\
        $(SOURCE_ROOT)/util/thread/queue.cpp\
        $(SOURCE_ROOT)/util/thread/lfqueue.cpp\
        $(SOURCE_ROOT)/util/thread/lfstack.cpp\
        $(SOURCE_ROOT)/util/thread/singleton.cpp\
        $(SOURCE_ROOT)/util/thread/fwd.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_thread.cpp' util/thread/pool.cpp util/thread/queue.cpp util/thread/lfqueue.cpp util/thread/lfstack.cpp util/thread/singleton.cpp util/thread/fwd.cpp

$(BUILD_ROOT)/util/all_system_2.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_system_2.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_system_2.cpp.o' '$(BUILD_ROOT)/util/all_system_2.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_system_2.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/system/madvise.cpp\
        $(SOURCE_ROOT)/util/system/mem_info.cpp\
        $(SOURCE_ROOT)/util/system/mktemp.cpp\
        $(SOURCE_ROOT)/util/system/mlock.cpp\
        $(SOURCE_ROOT)/util/system/mutex.cpp\
        $(SOURCE_ROOT)/util/system/nice.cpp\
        $(SOURCE_ROOT)/util/system/pipe.cpp\
        $(SOURCE_ROOT)/util/system/platform.cpp\
        $(SOURCE_ROOT)/util/system/progname.cpp\
        $(SOURCE_ROOT)/util/system/protect.cpp\
        $(SOURCE_ROOT)/util/system/rusage.cpp\
        $(SOURCE_ROOT)/util/system/rwlock.cpp\
        $(SOURCE_ROOT)/util/system/sanitizers.cpp\
        $(SOURCE_ROOT)/util/system/sem.cpp\
        $(SOURCE_ROOT)/util/system/shmat.cpp\
        $(SOURCE_ROOT)/util/system/spin_wait.cpp\
        $(SOURCE_ROOT)/util/system/spinlock.cpp\
        $(SOURCE_ROOT)/util/system/sysstat.cpp\
        $(SOURCE_ROOT)/util/system/sys_alloc.cpp\
        $(SOURCE_ROOT)/util/system/tempfile.cpp\
        $(SOURCE_ROOT)/util/system/thread.cpp\
        $(SOURCE_ROOT)/util/system/tls.cpp\
        $(SOURCE_ROOT)/util/system/types.cpp\
        $(SOURCE_ROOT)/util/system/user.cpp\
        $(SOURCE_ROOT)/util/system/yassert.cpp\
        $(SOURCE_ROOT)/util/system/yield.cpp\
        $(SOURCE_ROOT)/util/system/shellcommand.cpp\
        $(SOURCE_ROOT)/util/system/src_location.cpp\
        $(SOURCE_ROOT)/util/system/unaligned_mem.cpp\
        $(SOURCE_ROOT)/util/system/align.cpp\
        $(SOURCE_ROOT)/util/system/atomic.cpp\
        $(SOURCE_ROOT)/util/system/byteorder.cpp\
        $(SOURCE_ROOT)/util/system/fhandle.cpp\
        $(SOURCE_ROOT)/util/system/guard.cpp\
        $(SOURCE_ROOT)/util/system/maxlen.cpp\
        $(SOURCE_ROOT)/util/system/sigset.cpp\
        $(SOURCE_ROOT)/util/system/utime.cpp\
        $(SOURCE_ROOT)/util/system/cpu_id.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_system_2.cpp' util/system/madvise.cpp util/system/mem_info.cpp util/system/mktemp.cpp util/system/mlock.cpp util/system/mutex.cpp util/system/nice.cpp util/system/pipe.cpp util/system/platform.cpp util/system/progname.cpp util/system/protect.cpp util/system/rusage.cpp util/system/rwlock.cpp util/system/sanitizers.cpp util/system/sem.cpp util/system/shmat.cpp util/system/spin_wait.cpp util/system/spinlock.cpp util/system/sysstat.cpp util/system/sys_alloc.cpp util/system/tempfile.cpp util/system/thread.cpp util/system/tls.cpp util/system/types.cpp util/system/user.cpp util/system/yassert.cpp util/system/yield.cpp util/system/shellcommand.cpp util/system/src_location.cpp util/system/unaligned_mem.cpp util/system/align.cpp util/system/atomic.cpp util/system/byteorder.cpp util/system/fhandle.cpp util/system/guard.cpp util/system/maxlen.cpp util/system/sigset.cpp util/system/utime.cpp util/system/cpu_id.cpp

$(BUILD_ROOT)/util/all_system_1.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_system_1.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_system_1.cpp.o' '$(BUILD_ROOT)/util/all_system_1.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_system_1.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/system/atexit.cpp\
        $(SOURCE_ROOT)/util/system/backtrace.cpp\
        $(SOURCE_ROOT)/util/system/compat.cpp\
        $(SOURCE_ROOT)/util/system/compiler.cpp\
        $(SOURCE_ROOT)/util/system/condvar.cpp\
        $(SOURCE_ROOT)/util/system/context.cpp\
        $(SOURCE_ROOT)/util/system/daemon.cpp\
        $(SOURCE_ROOT)/util/system/datetime.cpp\
        $(SOURCE_ROOT)/util/system/defaults.c\
        $(SOURCE_ROOT)/util/system/demangle.cpp\
        $(SOURCE_ROOT)/util/system/direct_io.cpp\
        $(SOURCE_ROOT)/util/system/dynlib.cpp\
        $(SOURCE_ROOT)/util/system/env.cpp\
        $(SOURCE_ROOT)/util/system/err.cpp\
        $(SOURCE_ROOT)/util/system/error.cpp\
        $(SOURCE_ROOT)/util/system/event.cpp\
        $(SOURCE_ROOT)/util/system/execpath.cpp\
        $(SOURCE_ROOT)/util/system/fasttime.cpp\
        $(SOURCE_ROOT)/util/system/file.cpp\
        $(SOURCE_ROOT)/util/system/filemap.cpp\
        $(SOURCE_ROOT)/util/system/flock.cpp\
        $(SOURCE_ROOT)/util/system/file_lock.cpp\
        $(SOURCE_ROOT)/util/system/fs.cpp\
        $(SOURCE_ROOT)/util/system/fstat.cpp\
        $(SOURCE_ROOT)/util/system/getpid.cpp\
        $(SOURCE_ROOT)/util/system/hostname.cpp\
        $(SOURCE_ROOT)/util/system/hp_timer.cpp\
        $(SOURCE_ROOT)/util/system/info.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_system_1.cpp' util/system/atexit.cpp util/system/backtrace.cpp util/system/compat.cpp util/system/compiler.cpp util/system/condvar.cpp util/system/context.cpp util/system/daemon.cpp util/system/datetime.cpp util/system/defaults.c util/system/demangle.cpp util/system/direct_io.cpp util/system/dynlib.cpp util/system/env.cpp util/system/err.cpp util/system/error.cpp util/system/event.cpp util/system/execpath.cpp util/system/fasttime.cpp util/system/file.cpp util/system/filemap.cpp util/system/flock.cpp util/system/file_lock.cpp util/system/fs.cpp util/system/fstat.cpp util/system/getpid.cpp util/system/hostname.cpp util/system/hp_timer.cpp util/system/info.cpp

$(BUILD_ROOT)/util/all_string.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_string.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_string.cpp.o' '$(BUILD_ROOT)/util/all_string.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_string.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/string/builder.cpp\
        $(SOURCE_ROOT)/util/string/cgiparam.cpp\
        $(SOURCE_ROOT)/util/string/delim_string_iter.cpp\
        $(SOURCE_ROOT)/util/string/escape.cpp\
        $(SOURCE_ROOT)/util/string/util.cpp\
        $(SOURCE_ROOT)/util/string/vector.cpp\
        $(SOURCE_ROOT)/util/string/split_iterator.cpp\
        $(SOURCE_ROOT)/util/string/split.cpp\
        $(SOURCE_ROOT)/util/string/url.cpp\
        $(SOURCE_ROOT)/util/string/kmp.cpp\
        $(SOURCE_ROOT)/util/string/quote.cpp\
        $(SOURCE_ROOT)/util/string/ascii.cpp\
        $(SOURCE_ROOT)/util/string/printf.cpp\
        $(SOURCE_ROOT)/util/string/type.cpp\
        $(SOURCE_ROOT)/util/string/strip.cpp\
        $(SOURCE_ROOT)/util/string/pcdata.cpp\
        $(SOURCE_ROOT)/util/string/hex.cpp\
        $(SOURCE_ROOT)/util/string/cstriter.cpp\
        $(SOURCE_ROOT)/util/string/iterator.cpp\
        $(SOURCE_ROOT)/util/string/join.cpp\
        $(SOURCE_ROOT)/util/string/scan.cpp\
        $(SOURCE_ROOT)/util/string/strspn.cpp\
        $(SOURCE_ROOT)/util/string/subst.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_string.cpp' util/string/builder.cpp util/string/cgiparam.cpp util/string/delim_string_iter.cpp util/string/escape.cpp util/string/util.cpp util/string/vector.cpp util/string/split_iterator.cpp util/string/split.cpp util/string/url.cpp util/string/kmp.cpp util/string/quote.cpp util/string/ascii.cpp util/string/printf.cpp util/string/type.cpp util/string/strip.cpp util/string/pcdata.cpp util/string/hex.cpp util/string/cstriter.cpp util/string/iterator.cpp util/string/join.cpp util/string/scan.cpp util/string/strspn.cpp util/string/subst.cpp

$(BUILD_ROOT)/util/all_stream.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_stream.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_stream.cpp.o' '$(BUILD_ROOT)/util/all_stream.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_stream.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/stream/buffer.cpp\
        $(SOURCE_ROOT)/util/stream/buffered.cpp\
        $(SOURCE_ROOT)/util/stream/debug.cpp\
        $(SOURCE_ROOT)/util/stream/direct_io.cpp\
        $(SOURCE_ROOT)/util/stream/file.cpp\
        $(SOURCE_ROOT)/util/stream/hex.cpp\
        $(SOURCE_ROOT)/util/stream/input.cpp\
        $(SOURCE_ROOT)/util/stream/length.cpp\
        $(SOURCE_ROOT)/util/stream/mem.cpp\
        $(SOURCE_ROOT)/util/stream/multi.cpp\
        $(SOURCE_ROOT)/util/stream/null.cpp\
        $(SOURCE_ROOT)/util/stream/output.cpp\
        $(SOURCE_ROOT)/util/stream/pipe.cpp\
        $(SOURCE_ROOT)/util/stream/str.cpp\
        $(SOURCE_ROOT)/util/stream/tee.cpp\
        $(SOURCE_ROOT)/util/stream/zerocopy.cpp\
        $(SOURCE_ROOT)/util/stream/zlib.cpp\
        $(SOURCE_ROOT)/util/stream/printf.cpp\
        $(SOURCE_ROOT)/util/stream/format.cpp\
        $(SOURCE_ROOT)/util/stream/tempbuf.cpp\
        $(SOURCE_ROOT)/util/stream/walk.cpp\
        $(SOURCE_ROOT)/util/stream/aligned.cpp\
        $(SOURCE_ROOT)/util/stream/holder.cpp\
        $(SOURCE_ROOT)/util/stream/labeled.cpp\
        $(SOURCE_ROOT)/util/stream/tokenizer.cpp\
        $(SOURCE_ROOT)/util/stream/trace.cpp\
        $(SOURCE_ROOT)/util/stream/fwd.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_stream.cpp' util/stream/buffer.cpp util/stream/buffered.cpp util/stream/debug.cpp util/stream/direct_io.cpp util/stream/file.cpp util/stream/hex.cpp util/stream/input.cpp util/stream/length.cpp util/stream/mem.cpp util/stream/multi.cpp util/stream/null.cpp util/stream/output.cpp util/stream/pipe.cpp util/stream/str.cpp util/stream/tee.cpp util/stream/zerocopy.cpp util/stream/zlib.cpp util/stream/printf.cpp util/stream/format.cpp util/stream/tempbuf.cpp util/stream/walk.cpp util/stream/aligned.cpp util/stream/holder.cpp util/stream/labeled.cpp util/stream/tokenizer.cpp util/stream/trace.cpp util/stream/fwd.cpp

$(BUILD_ROOT)/util/all_random.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_random.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_random.cpp.o' '$(BUILD_ROOT)/util/all_random.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_random.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/random/common_ops.cpp\
        $(SOURCE_ROOT)/util/random/easy.cpp\
        $(SOURCE_ROOT)/util/random/fast.cpp\
        $(SOURCE_ROOT)/util/random/lcg_engine.cpp\
        $(SOURCE_ROOT)/util/random/entropy.cpp\
        $(SOURCE_ROOT)/util/random/mersenne.cpp\
        $(SOURCE_ROOT)/util/random/mersenne32.cpp\
        $(SOURCE_ROOT)/util/random/mersenne64.cpp\
        $(SOURCE_ROOT)/util/random/normal.cpp\
        $(SOURCE_ROOT)/util/random/shuffle.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_random.cpp' util/random/common_ops.cpp util/random/easy.cpp util/random/fast.cpp util/random/lcg_engine.cpp util/random/entropy.cpp util/random/mersenne.cpp util/random/mersenne32.cpp util/random/mersenne64.cpp util/random/normal.cpp util/random/shuffle.cpp

$(BUILD_ROOT)/util/all_network.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_network.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_network.cpp.o' '$(BUILD_ROOT)/util/all_network.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_network.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/network/hostip.cpp\
        $(SOURCE_ROOT)/util/network/init.cpp\
        $(SOURCE_ROOT)/util/network/poller.cpp\
        $(SOURCE_ROOT)/util/network/socket.cpp\
        $(SOURCE_ROOT)/util/network/pair.cpp\
        $(SOURCE_ROOT)/util/network/address.cpp\
        $(SOURCE_ROOT)/util/network/endpoint.cpp\
        $(SOURCE_ROOT)/util/network/interface.cpp\
        $(SOURCE_ROOT)/util/network/nonblock.cpp\
        $(SOURCE_ROOT)/util/network/iovec.cpp\
        $(SOURCE_ROOT)/util/network/ip.cpp\
        $(SOURCE_ROOT)/util/network/netloss.cpp\
        $(SOURCE_ROOT)/util/network/pollerimpl.cpp\
        $(SOURCE_ROOT)/util/network/sock.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_network.cpp' util/network/hostip.cpp util/network/init.cpp util/network/poller.cpp util/network/socket.cpp util/network/pair.cpp util/network/address.cpp util/network/endpoint.cpp util/network/interface.cpp util/network/nonblock.cpp util/network/iovec.cpp util/network/ip.cpp util/network/netloss.cpp util/network/pollerimpl.cpp util/network/sock.cpp

$(BUILD_ROOT)/util/all_memory.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_memory.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_memory.cpp.o' '$(BUILD_ROOT)/util/all_memory.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_memory.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/memory/tempbuf.cpp\
        $(SOURCE_ROOT)/util/memory/blob.cpp\
        $(SOURCE_ROOT)/util/memory/mmapalloc.cpp\
        $(SOURCE_ROOT)/util/memory/alloc.cpp\
        $(SOURCE_ROOT)/util/memory/pool.cpp\
        $(SOURCE_ROOT)/util/memory/addstorage.cpp\
        $(SOURCE_ROOT)/util/memory/segmented_string_pool.cpp\
        $(SOURCE_ROOT)/util/memory/segpool_alloc.cpp\
        $(SOURCE_ROOT)/util/memory/smallobj.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_memory.cpp' util/memory/tempbuf.cpp util/memory/blob.cpp util/memory/mmapalloc.cpp util/memory/alloc.cpp util/memory/pool.cpp util/memory/addstorage.cpp util/memory/segmented_string_pool.cpp util/memory/segpool_alloc.cpp util/memory/smallobj.cpp

$(BUILD_ROOT)/util/all_generic.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_generic.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_generic.cpp.o' '$(BUILD_ROOT)/util/all_generic.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_generic.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/generic/adaptor.cpp\
        $(SOURCE_ROOT)/util/generic/array_ref.cpp\
        $(SOURCE_ROOT)/util/generic/array_size.cpp\
        $(SOURCE_ROOT)/util/generic/buffer.cpp\
        $(SOURCE_ROOT)/util/generic/chartraits.cpp\
        $(SOURCE_ROOT)/util/generic/explicit_type.cpp\
        $(SOURCE_ROOT)/util/generic/function.cpp\
        $(SOURCE_ROOT)/util/generic/guid.cpp\
        $(SOURCE_ROOT)/util/generic/hash.cpp\
        $(SOURCE_ROOT)/util/generic/hash_primes.cpp\
        $(SOURCE_ROOT)/util/generic/hide_ptr.cpp\
        $(SOURCE_ROOT)/util/generic/mem_copy.cpp\
        $(SOURCE_ROOT)/util/generic/ptr.cpp\
        $(SOURCE_ROOT)/util/generic/singleton.cpp\
        $(SOURCE_ROOT)/util/generic/strbuf.cpp\
        $(SOURCE_ROOT)/util/generic/strfcpy.cpp\
        $(SOURCE_ROOT)/util/generic/string.cpp\
        $(SOURCE_ROOT)/util/generic/utility.cpp\
        $(SOURCE_ROOT)/util/generic/va_args.cpp\
        $(SOURCE_ROOT)/util/generic/xrange.cpp\
        $(SOURCE_ROOT)/util/generic/yexception.cpp\
        $(SOURCE_ROOT)/util/generic/ymath.cpp\
        $(SOURCE_ROOT)/util/generic/algorithm.cpp\
        $(SOURCE_ROOT)/util/generic/bitmap.cpp\
        $(SOURCE_ROOT)/util/generic/bitops.cpp\
        $(SOURCE_ROOT)/util/generic/bt_exception.cpp\
        $(SOURCE_ROOT)/util/generic/cast.cpp\
        $(SOURCE_ROOT)/util/generic/deque.cpp\
        $(SOURCE_ROOT)/util/generic/fastqueue.cpp\
        $(SOURCE_ROOT)/util/generic/flags.cpp\
        $(SOURCE_ROOT)/util/generic/fwd.cpp\
        $(SOURCE_ROOT)/util/generic/hash_set.cpp\
        $(SOURCE_ROOT)/util/generic/intrlist.cpp\
        $(SOURCE_ROOT)/util/generic/is_in.cpp\
        $(SOURCE_ROOT)/util/generic/iterator.cpp\
        $(SOURCE_ROOT)/util/generic/iterator_range.cpp\
        $(SOURCE_ROOT)/util/generic/lazy_value.cpp\
        $(SOURCE_ROOT)/util/generic/list.cpp\
        $(SOURCE_ROOT)/util/generic/map.cpp\
        $(SOURCE_ROOT)/util/generic/mapfindptr.cpp\
        $(SOURCE_ROOT)/util/generic/maybe.cpp\
        $(SOURCE_ROOT)/util/generic/noncopyable.cpp\
        $(SOURCE_ROOT)/util/generic/object_counter.cpp\
        $(SOURCE_ROOT)/util/generic/queue.cpp\
        $(SOURCE_ROOT)/util/generic/refcount.cpp\
        $(SOURCE_ROOT)/util/generic/region.cpp\
        $(SOURCE_ROOT)/util/generic/reinterpretcast.cpp\
        $(SOURCE_ROOT)/util/generic/set.cpp\
        $(SOURCE_ROOT)/util/generic/stack.cpp\
        $(SOURCE_ROOT)/util/generic/stlfwd.cpp\
        $(SOURCE_ROOT)/util/generic/store_policy.cpp\
        $(SOURCE_ROOT)/util/generic/type_name.cpp\
        $(SOURCE_ROOT)/util/generic/typelist.cpp\
        $(SOURCE_ROOT)/util/generic/typetraits.cpp\
        $(SOURCE_ROOT)/util/generic/vector.cpp\
        $(SOURCE_ROOT)/util/generic/vector_ops.cpp\
        $(SOURCE_ROOT)/util/generic/ylimits.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_generic.cpp' util/generic/adaptor.cpp util/generic/array_ref.cpp util/generic/array_size.cpp util/generic/buffer.cpp util/generic/chartraits.cpp util/generic/explicit_type.cpp util/generic/function.cpp util/generic/guid.cpp util/generic/hash.cpp util/generic/hash_primes.cpp util/generic/hide_ptr.cpp util/generic/mem_copy.cpp util/generic/ptr.cpp util/generic/singleton.cpp util/generic/strbuf.cpp util/generic/strfcpy.cpp util/generic/string.cpp util/generic/utility.cpp util/generic/va_args.cpp util/generic/xrange.cpp util/generic/yexception.cpp util/generic/ymath.cpp util/generic/algorithm.cpp util/generic/bitmap.cpp util/generic/bitops.cpp util/generic/bt_exception.cpp util/generic/cast.cpp util/generic/deque.cpp util/generic/fastqueue.cpp util/generic/flags.cpp util/generic/fwd.cpp util/generic/hash_set.cpp util/generic/intrlist.cpp util/generic/is_in.cpp util/generic/iterator.cpp util/generic/iterator_range.cpp util/generic/lazy_value.cpp util/generic/list.cpp util/generic/map.cpp util/generic/mapfindptr.cpp util/generic/maybe.cpp util/generic/noncopyable.cpp util/generic/object_counter.cpp util/generic/queue.cpp util/generic/refcount.cpp util/generic/region.cpp util/generic/reinterpretcast.cpp util/generic/set.cpp util/generic/stack.cpp util/generic/stlfwd.cpp util/generic/store_policy.cpp util/generic/type_name.cpp util/generic/typelist.cpp util/generic/typetraits.cpp util/generic/vector.cpp util/generic/vector_ops.cpp util/generic/ylimits.cpp

$(BUILD_ROOT)/util/all_folder.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_folder.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_folder.cpp.o' '$(BUILD_ROOT)/util/all_folder.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_folder.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/folder/fts.cpp\
        $(SOURCE_ROOT)/util/folder/filelist.cpp\
        $(SOURCE_ROOT)/util/folder/dirut.cpp\
        $(SOURCE_ROOT)/util/folder/path.cpp\
        $(SOURCE_ROOT)/util/folder/pathsplit.cpp\
        $(SOURCE_ROOT)/util/folder/iterator.cpp\
        $(SOURCE_ROOT)/util/folder/tempdir.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_folder.cpp' util/folder/fts.cpp util/folder/filelist.cpp util/folder/dirut.cpp util/folder/path.cpp util/folder/pathsplit.cpp util/folder/iterator.cpp util/folder/tempdir.cpp

$(BUILD_ROOT)/util/all_util.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_util.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_util.cpp.o' '$(BUILD_ROOT)/util/all_util.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_util.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/ysafeptr.cpp\
        $(SOURCE_ROOT)/util/ysaveload.cpp\
        $(SOURCE_ROOT)/util/str_stl.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_util.cpp' util/ysafeptr.cpp util/ysaveload.cpp util/str_stl.cpp

$(BUILD_ROOT)/util/all_digest.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_digest.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_digest.cpp.o' '$(BUILD_ROOT)/util/all_digest.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_digest.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/digest/murmur.cpp\
        $(SOURCE_ROOT)/util/digest/fnv.cpp\
        $(SOURCE_ROOT)/util/digest/iterator.cpp\
        $(SOURCE_ROOT)/util/digest/numeric.cpp\
        $(SOURCE_ROOT)/util/digest/multi.cpp\
        $(SOURCE_ROOT)/util/digest/sequence.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_digest.cpp' util/digest/murmur.cpp util/digest/fnv.cpp util/digest/iterator.cpp util/digest/numeric.cpp util/digest/multi.cpp util/digest/sequence.cpp

$(BUILD_ROOT)/util/all_datetime.cpp.o\
        ::\
        $(BUILD_ROOT)/util/all_datetime.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/util/all_datetime.cpp.o' '$(BUILD_ROOT)/util/all_datetime.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/util/all_datetime.cpp\
        ::\
        $(SOURCE_ROOT)/build/scripts/gen_join_srcs.py\
        $(SOURCE_ROOT)/util/datetime/base.cpp\
        $(SOURCE_ROOT)/util/datetime/cputimer.cpp\
        $(SOURCE_ROOT)/util/datetime/systime.cpp\
        $(SOURCE_ROOT)/util/datetime/constants.cpp\

	mkdir -p '$(BUILD_ROOT)/util'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/gen_join_srcs.py' '$(BUILD_ROOT)/util/all_datetime.cpp' util/datetime/base.cpp util/datetime/cputimer.cpp util/datetime/systime.cpp util/datetime/constants.cpp

$(BUILD_ROOT)/catboost/libs/cat_feature/libcatboost-libs-cat_feature.a\
$(BUILD_ROOT)/catboost/libs/cat_feature/libcatboost-libs-cat_feature.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/cat_feature/cat_feature.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/cat_feature'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-cat_feature -o catboost/libs/cat_feature/libcatboost-libs-cat_feature.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/cat_feature/libcatboost-libs-cat_feature.a' '$(BUILD_ROOT)/catboost/libs/cat_feature/cat_feature.cpp.o'

$(BUILD_ROOT)/catboost/libs/cat_feature/cat_feature.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/cat_feature/cat_feature.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/cat_feature'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/cat_feature/cat_feature.cpp.o' '$(SOURCE_ROOT)/catboost/libs/cat_feature/cat_feature.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/colorizer/liblibrary-colorizer.a\
$(BUILD_ROOT)/library/colorizer/liblibrary-colorizer.a.mf\
        ::\
        $(BUILD_ROOT)/library/colorizer/output.cpp.o\
        $(BUILD_ROOT)/library/colorizer/colors.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/colorizer'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-colorizer -o library/colorizer/liblibrary-colorizer.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/colorizer/liblibrary-colorizer.a' '$(BUILD_ROOT)/library/colorizer/output.cpp.o' '$(BUILD_ROOT)/library/colorizer/colors.cpp.o'

$(BUILD_ROOT)/library/colorizer/output.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/colorizer/output.cpp\

	mkdir -p '$(BUILD_ROOT)/library/colorizer'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/colorizer/output.cpp.o' '$(SOURCE_ROOT)/library/colorizer/output.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/colorizer/colors.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/colorizer/colors.cpp\

	mkdir -p '$(BUILD_ROOT)/library/colorizer'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/colorizer/colors.cpp.o' '$(SOURCE_ROOT)/library/colorizer/colors.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/liblibrary-getopt-small.a\
$(BUILD_ROOT)/library/getopt/small/liblibrary-getopt-small.a.mf\
        ::\
        $(BUILD_ROOT)/library/getopt/small/ygetopt.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/posix_getopt.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/opt2.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/opt.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/modchooser.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt_parse_result.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt_parser.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt_opts.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt_opt.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp.o\
        $(BUILD_ROOT)/library/getopt/small/last_getopt.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-getopt-small -o library/getopt/small/liblibrary-getopt-small.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/getopt/small/liblibrary-getopt-small.a' '$(BUILD_ROOT)/library/getopt/small/ygetopt.cpp.o' '$(BUILD_ROOT)/library/getopt/small/posix_getopt.cpp.o' '$(BUILD_ROOT)/library/getopt/small/opt2.cpp.o' '$(BUILD_ROOT)/library/getopt/small/opt.cpp.o' '$(BUILD_ROOT)/library/getopt/small/modchooser.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt_parse_result.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt_parser.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt_opts.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt_opt.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp.o' '$(BUILD_ROOT)/library/getopt/small/last_getopt.cpp.o'

$(BUILD_ROOT)/library/getopt/small/ygetopt.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/ygetopt.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/ygetopt.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/ygetopt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/posix_getopt.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/posix_getopt.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/posix_getopt.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/posix_getopt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/opt2.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/opt2.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/opt2.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/opt2.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/opt.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/opt.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/opt.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/opt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/modchooser.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/modchooser.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/modchooser.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/modchooser.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt_parse_result.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt_parse_result.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt_parse_result.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt_parse_result.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt_parser.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt_parser.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt_parser.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt_parser.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt_opts.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt_opts.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt_opts.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt_opts.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt_opt.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt_opt.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt_opt.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt_opt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt_easy_setup.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/getopt/small/last_getopt.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/getopt/small/last_getopt.cpp\

	mkdir -p '$(BUILD_ROOT)/library/getopt/small'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/getopt/small/last_getopt.cpp.o' '$(SOURCE_ROOT)/library/getopt/small/last_getopt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
$(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a.mf\
        ::\
        $(BUILD_ROOT)/library/lfalloc/lf_allocX64.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/lfalloc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-lfalloc -o library/lfalloc/liblibrary-lfalloc.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a' '$(BUILD_ROOT)/library/lfalloc/lf_allocX64.cpp.o'

$(BUILD_ROOT)/library/lfalloc/lf_allocX64.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/lfalloc/lf_allocX64.cpp\

	mkdir -p '$(BUILD_ROOT)/library/lfalloc'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/lfalloc/lf_allocX64.cpp.o' '$(SOURCE_ROOT)/library/lfalloc/lf_allocX64.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/library/logger/liblibrary-logger.a\
$(BUILD_ROOT)/library/logger/liblibrary-logger.a.mf\
        ::\
        $(BUILD_ROOT)/library/logger/filter.cpp.o\
        $(BUILD_ROOT)/library/logger/element.cpp.o\
        $(BUILD_ROOT)/library/logger/stream.cpp.o\
        $(BUILD_ROOT)/library/logger/thread.cpp.o\
        $(BUILD_ROOT)/library/logger/backend.cpp.o\
        $(BUILD_ROOT)/library/logger/null.cpp.o\
        $(BUILD_ROOT)/library/logger/file.cpp.o\
        $(BUILD_ROOT)/library/logger/system.cpp.o\
        $(BUILD_ROOT)/library/logger/log.cpp.o\
        $(BUILD_ROOT)/library/logger/priority.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-logger -o library/logger/liblibrary-logger.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/logger/liblibrary-logger.a' '$(BUILD_ROOT)/library/logger/filter.cpp.o' '$(BUILD_ROOT)/library/logger/element.cpp.o' '$(BUILD_ROOT)/library/logger/stream.cpp.o' '$(BUILD_ROOT)/library/logger/thread.cpp.o' '$(BUILD_ROOT)/library/logger/backend.cpp.o' '$(BUILD_ROOT)/library/logger/null.cpp.o' '$(BUILD_ROOT)/library/logger/file.cpp.o' '$(BUILD_ROOT)/library/logger/system.cpp.o' '$(BUILD_ROOT)/library/logger/log.cpp.o' '$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp.o'

$(BUILD_ROOT)/library/logger/filter.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/filter.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/filter.cpp.o' '$(SOURCE_ROOT)/library/logger/filter.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/element.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/element.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/element.cpp.o' '$(SOURCE_ROOT)/library/logger/element.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/stream.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/stream.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/stream.cpp.o' '$(SOURCE_ROOT)/library/logger/stream.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/thread.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/thread.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/thread.cpp.o' '$(SOURCE_ROOT)/library/logger/thread.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/backend.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/backend.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/backend.cpp.o' '$(SOURCE_ROOT)/library/logger/backend.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/null.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/null.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/null.cpp.o' '$(SOURCE_ROOT)/library/logger/null.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/file.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/file.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/file.cpp.o' '$(SOURCE_ROOT)/library/logger/file.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/system.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/system.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/system.cpp.o' '$(SOURCE_ROOT)/library/logger/system.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/log.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/log.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/log.cpp.o' '$(SOURCE_ROOT)/library/logger/log.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/library/logger/priority.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp.o' '$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/library/logger/priority.h\

	mkdir -p '$(BUILD_ROOT)/library/logger'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/library/logger/priority.h' --include-path library/logger/priority.h --output '$(BUILD_ROOT)/library/logger/priority.h_serialized.cpp'

$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/util/charset/libutil-charset.a\
        $(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a\
        $(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/util/libyutil.a\
        $(BUILD_ROOT)/library/colorizer/liblibrary-colorizer.a\
        $(BUILD_ROOT)/library/getopt/small/liblibrary-getopt-small.a\
        $(BUILD_ROOT)/library/cppparser/liblibrary-cppparser.a\
        $(BUILD_ROOT)/tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a\
        $(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/main.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/tools/enum_parser/enum_parser'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name enum_parser -o tools/enum_parser/enum_parser/enum_parser.mf -t PROGRAM -Ya,lics -Ya,peers library/getopt/small/liblibrary-getopt-small.a tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a library/colorizer/liblibrary-colorizer.a library/cppparser/liblibrary-cppparser.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/tools/enum_parser/enum_parser/main.cpp.o' -o '$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' -rdynamic -Wl,--start-group library/getopt/small/liblibrary-getopt-small.a tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a library/colorizer/liblibrary-colorizer.a library/cppparser/liblibrary-cppparser.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lrt -ldl -lpthread -nodefaultlibs -lpthread -lc -lm -s

$(BUILD_ROOT)/library/cppparser/liblibrary-cppparser.a\
$(BUILD_ROOT)/library/cppparser/liblibrary-cppparser.a.mf\
        ::\
        $(BUILD_ROOT)/library/cppparser/parser.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/cppparser'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-cppparser -o library/cppparser/liblibrary-cppparser.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/cppparser/liblibrary-cppparser.a' '$(BUILD_ROOT)/library/cppparser/parser.cpp.o'

$(BUILD_ROOT)/library/cppparser/parser.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/cppparser/parser.cpp\

	mkdir -p '$(BUILD_ROOT)/library/cppparser'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/cppparser/parser.cpp.o' '$(SOURCE_ROOT)/library/cppparser/parser.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a\
$(BUILD_ROOT)/tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a.mf\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/tools/enum_parser/parse_enum'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name tools-enum_parser-parse_enum -o tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/tools/enum_parser/parse_enum/libtools-enum_parser-parse_enum.a' '$(BUILD_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp.o'

$(BUILD_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp.o\
        ::\
        $(SOURCE_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp\

	mkdir -p '$(BUILD_ROOT)/tools/enum_parser/parse_enum'
	'$(CXX)' -c -o '$(BUILD_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp.o' '$(SOURCE_ROOT)/tools/enum_parser/parse_enum/parse_enum.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/tools/enum_parser/enum_parser/main.cpp.o\
        ::\
        $(SOURCE_ROOT)/tools/enum_parser/enum_parser/main.cpp\

	mkdir -p '$(BUILD_ROOT)/tools/enum_parser/enum_parser'
	'$(CXX)' -c -o '$(BUILD_ROOT)/tools/enum_parser/enum_parser/main.cpp.o' '$(SOURCE_ROOT)/tools/enum_parser/enum_parser/main.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/global/liblibrary-logger-global.a\
$(BUILD_ROOT)/library/logger/global/liblibrary-logger-global.a.mf\
        ::\
        $(BUILD_ROOT)/library/logger/global/rty_formater.cpp.o\
        $(BUILD_ROOT)/library/logger/global/global.cpp.o\
        $(BUILD_ROOT)/library/logger/global/common.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/logger/global'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-logger-global -o library/logger/global/liblibrary-logger-global.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/logger/global/liblibrary-logger-global.a' '$(BUILD_ROOT)/library/logger/global/rty_formater.cpp.o' '$(BUILD_ROOT)/library/logger/global/global.cpp.o' '$(BUILD_ROOT)/library/logger/global/common.cpp.o'

$(BUILD_ROOT)/library/logger/global/rty_formater.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/global/rty_formater.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger/global'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/global/rty_formater.cpp.o' '$(SOURCE_ROOT)/library/logger/global/rty_formater.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/global/global.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/global/global.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger/global'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/global/global.cpp.o' '$(SOURCE_ROOT)/library/logger/global/global.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/logger/global/common.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/logger/global/common.cpp\

	mkdir -p '$(BUILD_ROOT)/library/logger/global'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/logger/global/common.cpp.o' '$(SOURCE_ROOT)/library/logger/global/common.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/logging/libcatboost-libs-logging.a\
$(BUILD_ROOT)/catboost/libs/logging/libcatboost-libs-logging.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/logging/logging.cpp.o\
        $(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/logging'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-logging -o catboost/libs/logging/libcatboost-libs-logging.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/logging/libcatboost-libs-logging.a' '$(BUILD_ROOT)/catboost/libs/logging/logging.cpp.o' '$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/logging/logging.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/logging/logging.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/logging'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/logging/logging.cpp.o' '$(SOURCE_ROOT)/catboost/libs/logging/logging.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/logging'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/logging/logging_level.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/logging'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/logging/logging_level.h' --include-path catboost/libs/logging/logging_level.h --output '$(BUILD_ROOT)/catboost/libs/logging/logging_level.h_serialized.cpp'

$(BUILD_ROOT)/contrib/libs/rapidjson/libcontrib-libs-rapidjson.a\
$(BUILD_ROOT)/contrib/libs/rapidjson/libcontrib-libs-rapidjson.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/rapidjson/__/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/rapidjson'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-rapidjson -o contrib/libs/rapidjson/libcontrib-libs-rapidjson.a.mf -t LIBRARY -Ya,lics RAPIDJSON -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/rapidjson/libcontrib-libs-rapidjson.a' '$(BUILD_ROOT)/contrib/libs/rapidjson/__/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/contrib/libs/rapidjson/__/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/rapidjson/__/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/rapidjson/__/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/rapidjson/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/common/liblibrary-json-common.a\
$(BUILD_ROOT)/library/json/common/liblibrary-json-common.a.mf\
        ::\
        $(BUILD_ROOT)/library/json/common/defs.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/json/common'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-json-common -o library/json/common/liblibrary-json-common.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/json/common/liblibrary-json-common.a' '$(BUILD_ROOT)/library/json/common/defs.cpp.o'

$(BUILD_ROOT)/library/json/common/defs.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/common/defs.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/common'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/common/defs.cpp.o' '$(SOURCE_ROOT)/library/json/common/defs.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/writer/liblibrary-json-writer.a\
$(BUILD_ROOT)/library/json/writer/liblibrary-json-writer.a.mf\
        ::\
        $(BUILD_ROOT)/library/json/writer/json.cpp.o\
        $(BUILD_ROOT)/library/json/writer/json_value.cpp.o\
        $(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/json/writer'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-json-writer -o library/json/writer/liblibrary-json-writer.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/json/writer/liblibrary-json-writer.a' '$(BUILD_ROOT)/library/json/writer/json.cpp.o' '$(BUILD_ROOT)/library/json/writer/json_value.cpp.o' '$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp.o'

$(BUILD_ROOT)/library/json/writer/json.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/writer/json.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/writer'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/writer/json.cpp.o' '$(SOURCE_ROOT)/library/json/writer/json.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/writer/json_value.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/writer/json_value.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/writer'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/writer/json_value.cpp.o' '$(SOURCE_ROOT)/library/json/writer/json_value.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/writer'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp.o' '$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/library/json/writer/json_value.h\

	mkdir -p '$(BUILD_ROOT)/library/json/writer'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/library/json/writer/json_value.h' --include-path library/json/writer/json_value.h --output '$(BUILD_ROOT)/library/json/writer/json_value.h_serialized.cpp'

$(BUILD_ROOT)/library/json/fast_sax/liblibrary-json-fast_sax.a\
$(BUILD_ROOT)/library/json/fast_sax/liblibrary-json-fast_sax.a.mf\
        ::\
        $(BUILD_ROOT)/library/json/fast_sax/unescape.cpp.o\
        $(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/json/fast_sax'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-json-fast_sax -o library/json/fast_sax/liblibrary-json-fast_sax.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/json/fast_sax/liblibrary-json-fast_sax.a' '$(BUILD_ROOT)/library/json/fast_sax/unescape.cpp.o' '$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp.o'

$(BUILD_ROOT)/library/json/fast_sax/unescape.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/fast_sax/unescape.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/fast_sax'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/fast_sax/unescape.cpp.o' '$(SOURCE_ROOT)/library/json/fast_sax/unescape.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp.o\
        ::\
        $(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json/fast_sax'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp.o' '$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp\
        ::\
        $(BUILD_ROOT)/contrib/tools/ragel6/ragel6\
        $(SOURCE_ROOT)/library/json/fast_sax/parser.rl6\

	mkdir -p '$(BUILD_ROOT)/library/json/fast_sax'
	'$(BUILD_ROOT)/contrib/tools/ragel6/ragel6' -CT0 '-I$(SOURCE_ROOT)' -o '$(BUILD_ROOT)/library/json/fast_sax/parser.rl6.cpp' '$(SOURCE_ROOT)/library/json/fast_sax/parser.rl6'

$(BUILD_ROOT)/library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a\
$(BUILD_ROOT)/library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a.mf\
        ::\
        $(BUILD_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/string_utils/relaxed_escaper'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-string_utils-relaxed_escaper -o library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/string_utils/relaxed_escaper/liblibrary-string_utils-relaxed_escaper.a' '$(BUILD_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp.o'

$(BUILD_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp\

	mkdir -p '$(BUILD_ROOT)/library/string_utils/relaxed_escaper'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp.o' '$(SOURCE_ROOT)/library/string_utils/relaxed_escaper/relaxed_escaper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/liblibrary-json.a\
$(BUILD_ROOT)/library/json/liblibrary-json.a.mf\
        ::\
        $(BUILD_ROOT)/library/json/rapidjson_helpers.cpp.o\
        $(BUILD_ROOT)/library/json/json_prettifier.cpp.o\
        $(BUILD_ROOT)/library/json/json_reader.cpp.o\
        $(BUILD_ROOT)/library/json/json_writer.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/json'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-json -o library/json/liblibrary-json.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/json/liblibrary-json.a' '$(BUILD_ROOT)/library/json/rapidjson_helpers.cpp.o' '$(BUILD_ROOT)/library/json/json_prettifier.cpp.o' '$(BUILD_ROOT)/library/json/json_reader.cpp.o' '$(BUILD_ROOT)/library/json/json_writer.cpp.o'

$(BUILD_ROOT)/library/json/rapidjson_helpers.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/rapidjson_helpers.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/rapidjson_helpers.cpp.o' '$(SOURCE_ROOT)/library/json/rapidjson_helpers.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/json_prettifier.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/json_prettifier.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/json_prettifier.cpp.o' '$(SOURCE_ROOT)/library/json/json_prettifier.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/json_reader.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/json_reader.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/json_reader.cpp.o' '$(SOURCE_ROOT)/library/json/json_reader.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/json/json_writer.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/json/json_writer.cpp\

	mkdir -p '$(BUILD_ROOT)/library/json'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/json/json_writer.cpp.o' '$(SOURCE_ROOT)/library/json/json_writer.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/ctr_description/libcatboost-libs-ctr_description.a\
$(BUILD_ROOT)/catboost/libs/ctr_description/libcatboost-libs-ctr_description.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.cpp.o\
        $(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/ctr_description'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-ctr_description -o catboost/libs/ctr_description/libcatboost-libs-ctr_description.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/ctr_description/libcatboost-libs-ctr_description.a' '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.cpp.o' '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/ctr_description/ctr_type.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/ctr_description'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.cpp.o' '$(SOURCE_ROOT)/catboost/libs/ctr_description/ctr_type.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/ctr_description'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/ctr_description/ctr_type.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/ctr_description'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/ctr_description/ctr_type.h' --include-path catboost/libs/ctr_description/ctr_type.h --output '$(BUILD_ROOT)/catboost/libs/ctr_description/ctr_type.h_serialized.cpp'

$(BUILD_ROOT)/library/grid_creator/liblibrary-grid_creator.a\
$(BUILD_ROOT)/library/grid_creator/liblibrary-grid_creator.a.mf\
        ::\
        $(BUILD_ROOT)/library/grid_creator/median_in_bin_binarization.cpp.o\
        $(BUILD_ROOT)/library/grid_creator/binarization.cpp.o\
        $(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/grid_creator'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-grid_creator -o library/grid_creator/liblibrary-grid_creator.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/grid_creator/liblibrary-grid_creator.a' '$(BUILD_ROOT)/library/grid_creator/median_in_bin_binarization.cpp.o' '$(BUILD_ROOT)/library/grid_creator/binarization.cpp.o' '$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp.o'

$(BUILD_ROOT)/library/grid_creator/median_in_bin_binarization.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/grid_creator/median_in_bin_binarization.cpp\

	mkdir -p '$(BUILD_ROOT)/library/grid_creator'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/grid_creator/median_in_bin_binarization.cpp.o' '$(SOURCE_ROOT)/library/grid_creator/median_in_bin_binarization.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/grid_creator/binarization.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/grid_creator/binarization.cpp\

	mkdir -p '$(BUILD_ROOT)/library/grid_creator'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/grid_creator/binarization.cpp.o' '$(SOURCE_ROOT)/library/grid_creator/binarization.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/library/grid_creator'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp.o' '$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/library/grid_creator/binarization.h\

	mkdir -p '$(BUILD_ROOT)/library/grid_creator'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/library/grid_creator/binarization.h' --include-path library/grid_creator/binarization.h --output '$(BUILD_ROOT)/library/grid_creator/binarization.h_serialized.cpp'

$(BUILD_ROOT)/catboost/libs/options/libcatboost-libs-options.a\
$(BUILD_ROOT)/catboost/libs/options/libcatboost-libs-options.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/options/json_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/enum_helpers.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/output_file_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/oblivious_tree_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/metric_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/loss_description.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/system_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/data_processing_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/plain_options_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/catboost_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/cat_feature_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/boosting_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/bootstrap_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/overfitting_detector_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/binarization_options.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp.o\
        $(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-options -o catboost/libs/options/libcatboost-libs-options.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/options/libcatboost-libs-options.a' '$(BUILD_ROOT)/catboost/libs/options/json_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/enum_helpers.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/output_file_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/oblivious_tree_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/metric_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/loss_description.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/system_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/data_processing_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/plain_options_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/catboost_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/cat_feature_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/boosting_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/bootstrap_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/overfitting_detector_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/binarization_options.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/options/json_helper.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/json_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/json_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/json_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/enum_helpers.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/enum_helpers.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/enum_helpers.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/enum_helpers.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/output_file_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/output_file_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/output_file_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/output_file_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/oblivious_tree_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/oblivious_tree_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/oblivious_tree_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/oblivious_tree_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/metric_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/metric_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/metric_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/metric_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/loss_description.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/loss_description.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/loss_description.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/loss_description.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/system_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/system_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/system_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/system_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/data_processing_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/data_processing_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/data_processing_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/data_processing_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/plain_options_helper.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/plain_options_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/plain_options_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/plain_options_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/catboost_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/catboost_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/catboost_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/catboost_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/cat_feature_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/cat_feature_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/cat_feature_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/cat_feature_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/boosting_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/boosting_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/boosting_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/boosting_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/bootstrap_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/bootstrap_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/bootstrap_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/bootstrap_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/overfitting_detector_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/overfitting_detector_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/overfitting_detector_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/overfitting_detector_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/binarization_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/options/binarization_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/binarization_options.cpp.o' '$(SOURCE_ROOT)/catboost/libs/options/binarization_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/options/json_helper.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/options/json_helper.h' --include-path catboost/libs/options/json_helper.h --output '$(BUILD_ROOT)/catboost/libs/options/json_helper.h_serialized.cpp'

$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/options/enums.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/options'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/options/enums.h' --include-path catboost/libs/options/enums.h --output '$(BUILD_ROOT)/catboost/libs/options/enums.h_serialized.cpp'

$(BUILD_ROOT)/library/containers/2d_array/liblibrary-containers-2d_array.a\
$(BUILD_ROOT)/library/containers/2d_array/liblibrary-containers-2d_array.a.mf\
        ::\
        $(BUILD_ROOT)/library/containers/2d_array/2d_array.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/containers/2d_array'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-containers-2d_array -o library/containers/2d_array/liblibrary-containers-2d_array.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/containers/2d_array/liblibrary-containers-2d_array.a' '$(BUILD_ROOT)/library/containers/2d_array/2d_array.cpp.o'

$(BUILD_ROOT)/library/containers/2d_array/2d_array.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/containers/2d_array/2d_array.cpp\

	mkdir -p '$(BUILD_ROOT)/library/containers/2d_array'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/containers/2d_array/2d_array.cpp.o' '$(SOURCE_ROOT)/library/containers/2d_array/2d_array.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/binsaver/liblibrary-binsaver.a\
$(BUILD_ROOT)/library/binsaver/liblibrary-binsaver.a.mf\
        ::\
        $(BUILD_ROOT)/library/binsaver/util_stream_io.cpp.o\
        $(BUILD_ROOT)/library/binsaver/mem_io.cpp.o\
        $(BUILD_ROOT)/library/binsaver/buffered_io.cpp.o\
        $(BUILD_ROOT)/library/binsaver/blob_io.cpp.o\
        $(BUILD_ROOT)/library/binsaver/bin_saver.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-binsaver -o library/binsaver/liblibrary-binsaver.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/binsaver/liblibrary-binsaver.a' '$(BUILD_ROOT)/library/binsaver/util_stream_io.cpp.o' '$(BUILD_ROOT)/library/binsaver/mem_io.cpp.o' '$(BUILD_ROOT)/library/binsaver/buffered_io.cpp.o' '$(BUILD_ROOT)/library/binsaver/blob_io.cpp.o' '$(BUILD_ROOT)/library/binsaver/bin_saver.cpp.o'

$(BUILD_ROOT)/library/binsaver/util_stream_io.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/binsaver/util_stream_io.cpp\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/binsaver/util_stream_io.cpp.o' '$(SOURCE_ROOT)/library/binsaver/util_stream_io.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/binsaver/mem_io.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/binsaver/mem_io.cpp\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/binsaver/mem_io.cpp.o' '$(SOURCE_ROOT)/library/binsaver/mem_io.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/binsaver/buffered_io.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/binsaver/buffered_io.cpp\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/binsaver/buffered_io.cpp.o' '$(SOURCE_ROOT)/library/binsaver/buffered_io.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/binsaver/blob_io.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/binsaver/blob_io.cpp\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/binsaver/blob_io.cpp.o' '$(SOURCE_ROOT)/library/binsaver/blob_io.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/binsaver/bin_saver.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/binsaver/bin_saver.cpp\

	mkdir -p '$(BUILD_ROOT)/library/binsaver'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/binsaver/bin_saver.cpp.o' '$(SOURCE_ROOT)/library/binsaver/bin_saver.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a\
$(BUILD_ROOT)/contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/nayuki_md5'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-nayuki_md5 -o contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a.mf -t LIBRARY -Ya,lics MIT -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/nayuki_md5/libcontrib-libs-nayuki_md5.a' '$(BUILD_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S.o'

$(BUILD_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/nayuki_md5'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S.o' '$(SOURCE_ROOT)/contrib/libs/nayuki_md5/md5-fast-x8664.S' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include'

$(BUILD_ROOT)/contrib/libs/base64/avx2/liblibs-base64-avx2.a\
$(BUILD_ROOT)/contrib/libs/base64/avx2/liblibs-base64-avx2.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/avx2/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/avx2/codec_avx2.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/avx2'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-avx2 -o contrib/libs/base64/avx2/liblibs-base64-avx2.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/avx2/liblibs-base64-avx2.a' '$(BUILD_ROOT)/contrib/libs/base64/avx2/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/avx2/codec_avx2.c.o'

$(BUILD_ROOT)/contrib/libs/base64/avx2/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/avx2/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/avx2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/avx2/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/avx2/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -mavx2 -std=c11

$(BUILD_ROOT)/contrib/libs/base64/avx2/codec_avx2.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/avx2/codec_avx2.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/avx2'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/avx2/codec_avx2.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/avx2/codec_avx2.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -mavx2 -std=c11

$(BUILD_ROOT)/contrib/libs/base64/ssse3/liblibs-base64-ssse3.a\
$(BUILD_ROOT)/contrib/libs/base64/ssse3/liblibs-base64-ssse3.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/ssse3/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/ssse3'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-ssse3 -o contrib/libs/base64/ssse3/liblibs-base64-ssse3.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/ssse3/liblibs-base64-ssse3.a' '$(BUILD_ROOT)/contrib/libs/base64/ssse3/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c.o'

$(BUILD_ROOT)/contrib/libs/base64/ssse3/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/ssse3/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/ssse3'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/ssse3/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/ssse3/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -mssse3 -std=c11

$(BUILD_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/ssse3'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/ssse3/codec_ssse3.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -mssse3 -std=c11

$(BUILD_ROOT)/contrib/libs/base64/neon32/liblibs-base64-neon32.a\
$(BUILD_ROOT)/contrib/libs/base64/neon32/liblibs-base64-neon32.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/neon32/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/neon32/codec_neon32.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon32'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-neon32 -o contrib/libs/base64/neon32/liblibs-base64-neon32.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/neon32/liblibs-base64-neon32.a' '$(BUILD_ROOT)/contrib/libs/base64/neon32/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/neon32/codec_neon32.c.o'

$(BUILD_ROOT)/contrib/libs/base64/neon32/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/neon32/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon32'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/neon32/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/neon32/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/contrib/libs/base64/neon32/codec_neon32.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/neon32/codec_neon32.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon32'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/neon32/codec_neon32.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/neon32/codec_neon32.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/contrib/libs/base64/neon64/liblibs-base64-neon64.a\
$(BUILD_ROOT)/contrib/libs/base64/neon64/liblibs-base64-neon64.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/neon64/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/neon64/codec_neon64.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon64'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-neon64 -o contrib/libs/base64/neon64/liblibs-base64-neon64.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/neon64/liblibs-base64-neon64.a' '$(BUILD_ROOT)/contrib/libs/base64/neon64/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/neon64/codec_neon64.c.o'

$(BUILD_ROOT)/contrib/libs/base64/neon64/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/neon64/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon64'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/neon64/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/neon64/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)'

$(BUILD_ROOT)/contrib/libs/base64/neon64/codec_neon64.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/neon64/codec_neon64.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/neon64'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/neon64/codec_neon64.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/neon64/codec_neon64.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)'

$(BUILD_ROOT)/contrib/libs/base64/plain32/liblibs-base64-plain32.a\
$(BUILD_ROOT)/contrib/libs/base64/plain32/liblibs-base64-plain32.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/plain32/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/plain32/codec_plain.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain32'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-plain32 -o contrib/libs/base64/plain32/liblibs-base64-plain32.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/plain32/liblibs-base64-plain32.a' '$(BUILD_ROOT)/contrib/libs/base64/plain32/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/plain32/codec_plain.c.o'

$(BUILD_ROOT)/contrib/libs/base64/plain32/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/plain32/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain32'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/plain32/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/plain32/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/contrib/libs/base64/plain32/codec_plain.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/plain32/codec_plain.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain32'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/plain32/codec_plain.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/plain32/codec_plain.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/contrib/libs/base64/plain64/liblibs-base64-plain64.a\
$(BUILD_ROOT)/contrib/libs/base64/plain64/liblibs-base64-plain64.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/base64/plain64/lib.c.o\
        $(BUILD_ROOT)/contrib/libs/base64/plain64/codec_plain.c.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain64'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-base64-plain64 -o contrib/libs/base64/plain64/liblibs-base64-plain64.a.mf -t LIBRARY -Ya,lics BSD2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/base64/plain64/liblibs-base64-plain64.a' '$(BUILD_ROOT)/contrib/libs/base64/plain64/lib.c.o' '$(BUILD_ROOT)/contrib/libs/base64/plain64/codec_plain.c.o'

$(BUILD_ROOT)/contrib/libs/base64/plain64/lib.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/plain64/lib.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain64'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/plain64/lib.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/plain64/lib.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/contrib/libs/base64/plain64/codec_plain.c.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/base64/plain64/codec_plain.c\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/base64/plain64'
	'$(CC)' -c -o '$(BUILD_ROOT)/contrib/libs/base64/plain64/codec_plain.c.o' '$(SOURCE_ROOT)/contrib/libs/base64/plain64/codec_plain.c' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -std=c11

$(BUILD_ROOT)/library/string_utils/base64/liblibrary-string_utils-base64.a\
$(BUILD_ROOT)/library/string_utils/base64/liblibrary-string_utils-base64.a.mf\
        ::\
        $(BUILD_ROOT)/library/string_utils/base64/base64.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/string_utils/base64'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-string_utils-base64 -o library/string_utils/base64/liblibrary-string_utils-base64.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/string_utils/base64/liblibrary-string_utils-base64.a' '$(BUILD_ROOT)/library/string_utils/base64/base64.cpp.o'

$(BUILD_ROOT)/library/string_utils/base64/base64.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/string_utils/base64/base64.cpp\

	mkdir -p '$(BUILD_ROOT)/library/string_utils/base64'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/string_utils/base64/base64.cpp.o' '$(SOURCE_ROOT)/library/string_utils/base64/base64.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/digest/md5/liblibrary-digest-md5.a\
$(BUILD_ROOT)/library/digest/md5/liblibrary-digest-md5.a.mf\
        ::\
        $(BUILD_ROOT)/library/digest/md5/md5.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/digest/md5'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-digest-md5 -o library/digest/md5/liblibrary-digest-md5.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/digest/md5/liblibrary-digest-md5.a' '$(BUILD_ROOT)/library/digest/md5/md5.cpp.o'

$(BUILD_ROOT)/library/digest/md5/md5.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/digest/md5/md5.cpp\

	mkdir -p '$(BUILD_ROOT)/library/digest/md5'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/digest/md5/md5.cpp.o' '$(SOURCE_ROOT)/library/digest/md5/md5.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/libcontrib-libs-crcutil.a\
$(BUILD_ROOT)/contrib/libs/crcutil/libcontrib-libs-crcutil.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc.o\
        $(BUILD_ROOT)/contrib/libs/crcutil/interface.cc.o\
        $(BUILD_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc.o\
        $(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc.o\
        $(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc.o\
        $(BUILD_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-crcutil -o contrib/libs/crcutil/libcontrib-libs-crcutil.a.mf -t LIBRARY -Ya,lics APACHE2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/crcutil/libcontrib-libs-crcutil.a' '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc.o' '$(BUILD_ROOT)/contrib/libs/crcutil/interface.cc.o' '$(BUILD_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc.o' '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc.o' '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc.o' '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc.o'

$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_intrinsic_i386_mmx.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/interface.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/interface.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/interface.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/interface.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/crc32c_sse4.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_i386_mmx.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/multiword_64_64_gcc_amd64_asm.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/crcutil'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc.o' '$(SOURCE_ROOT)/contrib/libs/crcutil/multiword_128_64_gcc_amd64_sse2.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DCRCUTIL_USE_MM_CRC32=1 -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/library/digest/crc32c/liblibrary-digest-crc32c.a\
$(BUILD_ROOT)/library/digest/crc32c/liblibrary-digest-crc32c.a.mf\
        ::\
        $(BUILD_ROOT)/library/digest/crc32c/crc32c.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/digest/crc32c'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-digest-crc32c -o library/digest/crc32c/liblibrary-digest-crc32c.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/digest/crc32c/liblibrary-digest-crc32c.a' '$(BUILD_ROOT)/library/digest/crc32c/crc32c.cpp.o'

$(BUILD_ROOT)/library/digest/crc32c/crc32c.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/digest/crc32c/crc32c.cpp\

	mkdir -p '$(BUILD_ROOT)/library/digest/crc32c'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/digest/crc32c/crc32c.cpp.o' '$(SOURCE_ROOT)/library/digest/crc32c/crc32c.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/threading/local_executor/liblibrary-threading-local_executor.a\
$(BUILD_ROOT)/library/threading/local_executor/liblibrary-threading-local_executor.a.mf\
        ::\
        $(BUILD_ROOT)/library/threading/local_executor/local_executor.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/threading/local_executor'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-threading-local_executor -o library/threading/local_executor/liblibrary-threading-local_executor.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/threading/local_executor/liblibrary-threading-local_executor.a' '$(BUILD_ROOT)/library/threading/local_executor/local_executor.cpp.o'

$(BUILD_ROOT)/library/threading/local_executor/local_executor.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/threading/local_executor/local_executor.cpp\

	mkdir -p '$(BUILD_ROOT)/library/threading/local_executor'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/threading/local_executor/local_executor.cpp.o' '$(SOURCE_ROOT)/library/threading/local_executor/local_executor.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/libcatboost-libs-helpers.a\
$(BUILD_ROOT)/catboost/libs/helpers/libcatboost-libs-helpers.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/helpers/data_split.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/query_info_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/binarize_target.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/restorable_rng.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/permutation.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/interrupt.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/matrix.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/progress_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/power_hash.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/dense_hash_view.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/dense_hash.cpp.o\
        $(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-helpers -o catboost/libs/helpers/libcatboost-libs-helpers.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/helpers/libcatboost-libs-helpers.a' '$(BUILD_ROOT)/catboost/libs/helpers/data_split.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/query_info_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/binarize_target.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/restorable_rng.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/permutation.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/interrupt.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/matrix.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/progress_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/power_hash.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/dense_hash_view.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/dense_hash.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/helpers/data_split.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/data_split.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/data_split.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/data_split.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/query_info_helper.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/query_info_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/query_info_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/query_info_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/binarize_target.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/binarize_target.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/binarize_target.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/binarize_target.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/restorable_rng.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/restorable_rng.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/restorable_rng.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/restorable_rng.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/permutation.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/permutation.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/permutation.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/permutation.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/eval_helpers.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/eval_helpers.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/interrupt.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/interrupt.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/interrupt.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/interrupt.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/matrix.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/matrix.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/matrix.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/matrix.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/progress_helper.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/progress_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/progress_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/progress_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/power_hash.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/power_hash.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/power_hash.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/power_hash.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/dense_hash_view.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/dense_hash_view.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/dense_hash_view.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/dense_hash_view.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/dense_hash.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/helpers/dense_hash.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/dense_hash.cpp.o' '$(SOURCE_ROOT)/catboost/libs/helpers/dense_hash.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/helpers/eval_helpers.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/helpers'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/helpers/eval_helpers.h' --include-path catboost/libs/helpers/eval_helpers.h --output '$(BUILD_ROOT)/catboost/libs/helpers/eval_helpers.h_serialized.cpp'

$(BUILD_ROOT)/catboost/libs/column_description/libcatboost-libs-column_description.a\
$(BUILD_ROOT)/catboost/libs/column_description/libcatboost-libs-column_description.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/column_description/cd_parser.cpp.o\
        $(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/column_description'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-column_description -o catboost/libs/column_description/libcatboost-libs-column_description.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/column_description/libcatboost-libs-column_description.a' '$(BUILD_ROOT)/catboost/libs/column_description/cd_parser.cpp.o' '$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/column_description/cd_parser.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/column_description/cd_parser.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/column_description'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/column_description/cd_parser.cpp.o' '$(SOURCE_ROOT)/catboost/libs/column_description/cd_parser.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/column_description'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/column_description/column.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/column_description'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/column_description/column.h' --include-path catboost/libs/column_description/column.h --output '$(BUILD_ROOT)/catboost/libs/column_description/column.h_serialized.cpp'

$(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a\
$(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/protobuf/wrappers.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/wire_format_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/wire_format.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/time_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/message_differencer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/json_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/utility.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/field_comparator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/unknown_field_set.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/type.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/timestamp.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/text_format.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/time.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/substitute.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/strutil.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/statusor.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/status.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/once.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/int128.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/common.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/struct.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/source_context.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/service.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/repeated_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/reflection_ops.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/messagext_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/messagext.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/message_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/message.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/map_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/json_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/tokenizer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/strtod.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/printer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/io/coded_stream.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/generated_message_util.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/field_mask.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/extension_set.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/empty.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/dynamic_message.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/duration.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/descriptor_database.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/descriptor.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/descriptor.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/compiler/parser.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/compiler/importer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/arenastring.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/arena.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/api.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/any.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/any.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-protobuf -o contrib/libs/protobuf/libcontrib-libs-protobuf.a.mf -t LIBRARY -Ya,lics BSD3 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a' '$(BUILD_ROOT)/contrib/libs/protobuf/wrappers.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/wire_format_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/wire_format.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/time_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/message_differencer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/json_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/utility.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/field_comparator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/unknown_field_set.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/type.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/timestamp.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/text_format.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/time.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/substitute.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/strutil.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/statusor.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/status.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/once.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/int128.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/common.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/struct.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/source_context.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/service.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/repeated_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/reflection_ops.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/messagext_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/messagext.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/message_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/message.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/map_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/json_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/tokenizer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/strtod.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/printer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/io/coded_stream.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_util.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/field_mask.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/extension_set.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/empty.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/dynamic_message.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/duration.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor_database.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/compiler/parser.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/compiler/importer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/arenastring.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/arena.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/api.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/any.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/any.cc.o'

$(BUILD_ROOT)/contrib/libs/protobuf/wrappers.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/wrappers.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/wrappers.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/wrappers.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/wire_format_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/wire_format_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/wire_format_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/wire_format_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/wire_format.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/wire_format.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/wire_format.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/wire_format.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/type_resolver_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/time_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/time_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/time_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/time_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/message_differencer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/message_differencer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/message_differencer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/message_differencer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/json_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/json_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/json_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/json_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/utility.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/utility.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/utility.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/utility.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/type_info.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectwriter.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/protostream_objectsource.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/proto_writer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/object_writer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_stream_parser.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_objectwriter.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/json_escaping.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/field_mask_utility.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/error_listener.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/default_value_objectwriter.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/internal/datapiece.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/field_mask_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/field_comparator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/field_comparator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/field_comparator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/field_comparator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/util'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/util/delimited_message_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/unknown_field_set.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/unknown_field_set.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/unknown_field_set.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/unknown_field_set.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/type.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/type.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/type.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/type.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/timestamp.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/timestamp.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/timestamp.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/timestamp.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/text_format.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/text_format.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/text_format.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/text_format.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/time.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/time.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/time.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/time.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/substitute.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/substitute.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/substitute.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/substitute.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/strutil.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/strutil.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/strutil.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/strutil.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/structurally_valid.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/stringprintf.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/stringpiece.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/statusor.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/statusor.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/statusor.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/statusor.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/status.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/status.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/status.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/status.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/once.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/once.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/once.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/once.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/mathlimits.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/int128.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/int128.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/int128.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/int128.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/common.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/common.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/common.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/common.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/bytestream.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_msvc.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/stubs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/stubs/atomicops_internals_x86_gcc.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/struct.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/struct.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/struct.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/struct.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/source_context.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/source_context.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/source_context.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/source_context.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/service.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/service.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/service.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/service.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/repeated_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/repeated_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/repeated_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/repeated_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/reflection_ops.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/reflection_ops.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/reflection_ops.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/reflection_ops.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/messagext_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/messagext_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/messagext_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/messagext_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/messagext.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/messagext.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/messagext.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/messagext.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/message_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/message_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/message_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/message_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/message.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/message.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/message.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/message.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/map_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/map_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/map_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/map_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/json_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/json_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/json_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/json_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream_impl.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/zero_copy_stream.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/tokenizer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/tokenizer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/tokenizer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/tokenizer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/strtod.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/strtod.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/strtod.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/strtod.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/printer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/printer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/printer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/printer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/gzip_stream.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/io/coded_stream.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/io/coded_stream.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/io'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/io/coded_stream.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/io/coded_stream.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_util.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_util.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_util.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_util.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_table_driven.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/generated_message_reflection.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/field_mask.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/field_mask.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/field_mask.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/field_mask.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/extension_set_heavy.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/extension_set.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/extension_set.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/extension_set.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/extension_set.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/empty.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/empty.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/empty.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/empty.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/dynamic_message.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/dynamic_message.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/dynamic_message.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/dynamic_message.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/duration.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/duration.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/duration.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/duration.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/descriptor_database.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/descriptor_database.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor_database.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/descriptor_database.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/descriptor.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/descriptor.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/descriptor.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/descriptor.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/descriptor.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/compiler/parser.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/parser.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/compiler/parser.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/parser.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/compiler/importer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/importer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/compiler/importer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/importer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/arenastring.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/arenastring.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/arenastring.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/arenastring.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/arena.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/arena.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/arena.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/arena.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/api.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/api.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/api.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/api.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/any.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/any.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/any.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/any.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/any.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/any.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/any.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/any.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DHAVE_ZLIB -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/libcontrib-libs-coreml.a\
$(BUILD_ROOT)/contrib/libs/coreml/libcontrib-libs-coreml.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-coreml -o contrib/libs/coreml/libcontrib-libs-coreml.a.mf -t LIBRARY -Ya,lics BSD -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/coreml/libcontrib-libs-coreml.a' '$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc.o'

$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/FeatureTypes.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/FeatureTypes.proto

$(BUILD_ROOT)/contrib/tools/protoc/protoc\
$(BUILD_ROOT)/contrib/tools/protoc/protoc.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/util/charset/libutil-charset.a\
        $(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a\
        $(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/util/libyutil.a\
        $(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
        $(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a\
        $(BUILD_ROOT)/contrib/tools/protoc/__/__/libs/protobuf/compiler/main.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/protoc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name protoc -o contrib/tools/protoc/protoc.mf -t PROGRAM -Ya,lics -Ya,peers contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/contrib/tools/protoc/__/__/libs/protobuf/compiler/main.cc.o' -o '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -rdynamic -Wl,--start-group contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lrt -ldl -lpthread -nodefaultlibs -lpthread -lc -lm -s

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a\
$(BUILD_ROOT)/contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/zip_writer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/subprocess.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/python/python_generator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/profile.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_helpers.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_generator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/parser.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/main.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_shared_code_generator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_service.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_name_resolver.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_helpers.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator_factory.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_file.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field_lite.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_doc_comment.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_context.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/importer.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_string_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_service.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_primitive_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_map_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_helpers.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_generator.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_file.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_extension.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum_field.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/command_line_interface.cc.o\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/code_generator.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-protobuf-protoc -o contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a.mf -t LIBRARY -Ya,lics BSD3 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/zip_writer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/subprocess.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/python/python_generator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/profile.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_helpers.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_generator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/parser.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/main.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_shared_code_generator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_service.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_name_resolver.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_helpers.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator_factory.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_file.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field_lite.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_doc_comment.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_context.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/importer.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_string_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_service.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_primitive_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_map_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_helpers.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_generator.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_file.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_extension.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum_field.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/command_line_interface.cc.o' '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/code_generator.cc.o'

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/zip_writer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/zip_writer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/zip_writer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/zip_writer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/subprocess.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/subprocess.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/subprocess.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/subprocess.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/python/python_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/python/python_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/python'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/python/python_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/python/python_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/profile.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/profile.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/profile.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/profile.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.pb.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/plugin.pb.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.pb.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/plugin.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/plugin.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/plugin.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/plugin.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_helpers.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/perlxs/perlxs_helpers.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_helpers.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/perlxs/perlxs_helpers.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/perlxs/perlxs_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/perlxs/perlxs_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/perlxs/perlxs_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/parser.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/parser.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/parser.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/parser.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/main.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/main.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/main.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/main.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_string_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_string_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_string_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_string_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_string_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_shared_code_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_shared_code_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_shared_code_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_shared_code_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_service.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_service.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_service.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_service.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_primitive_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_primitive_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_primitive_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_primitive_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_primitive_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_name_resolver.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_name_resolver.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_name_resolver.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_name_resolver.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_builder_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_builder_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_builder.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message_builder.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message_builder.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_message.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_message.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_map_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_map_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_map_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_map_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_map_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_lazy_message_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_lazy_message_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_lazy_message_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_lazy_message_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_lazy_message_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_helpers.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_helpers.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_helpers.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_helpers.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator_factory.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_generator_factory.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator_factory.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_generator_factory.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_file.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_file.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_file.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_file.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_extension_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_extension_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_extension.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_extension.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_extension.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field_lite.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_field_lite.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field_lite.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_field_lite.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_enum.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_enum.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_doc_comment.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_doc_comment.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_doc_comment.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_doc_comment.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_context.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_context.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/java/java_context.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/java/java_context.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/importer.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/importer.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/importer.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/importer.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_string_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_string_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_string_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_string_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_service.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_service.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_service.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_service.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_primitive_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_primitive_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_primitive_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_primitive_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_message_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_message_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_message.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_message.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_message.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_map_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_map_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_map_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_map_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_helpers.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_helpers.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_helpers.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_helpers.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_file.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_file.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_file.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_file.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_extension.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_extension.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_extension.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_extension.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum_field.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_enum_field.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum_field.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_enum_field.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_enum.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/cpp/cpp_enum.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/cpp/cpp_enum.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/command_line_interface.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/command_line_interface.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/command_line_interface.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/command_line_interface.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/code_generator.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/code_generator.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/protobuf/protoc/__/compiler/code_generator.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/code_generator.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/protoc/__/__/libs/protobuf/compiler/main.cc.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/protobuf/compiler/main.cc\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/protoc/__/__/libs/protobuf/compiler'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/protoc/__/__/libs/protobuf/compiler/main.cc.o' '$(SOURCE_ROOT)/contrib/libs/protobuf/compiler/main.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/util/charset/libutil-charset.a\
        $(BUILD_ROOT)/contrib/libs/zlib/libcontrib-libs-zlib.a\
        $(BUILD_ROOT)/contrib/libs/double-conversion/libcontrib-libs-double-conversion.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/util/libyutil.a\
        $(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
        $(BUILD_ROOT)/contrib/libs/protobuf/libcontrib-libs-protobuf.a\
        $(BUILD_ROOT)/contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name cpp_styleguide -o contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.mf -t PROGRAM -Ya,lics -Ya,peers contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp.o' -o '$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' -rdynamic -Wl,--start-group contrib/libs/protobuf/libcontrib-libs-protobuf.a contrib/libs/protobuf/protoc/liblibs-protobuf-protoc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a util/libyutil.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/zlib/libcontrib-libs-zlib.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a util/charset/libutil-charset.a contrib/libs/double-conversion/libcontrib-libs-double-conversion.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lrt -ldl -lpthread -nodefaultlibs -lpthread -lc -lm -s

$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp.o' '$(SOURCE_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -w -Wno-shadow -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/DataStructures.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/DataStructures.proto

$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/TreeEnsemble.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/TreeEnsemble.proto

$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/Scaler.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/Scaler.proto

$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/SVM.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/SVM.proto

$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/OneHotEncoder.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/OneHotEncoder.proto

$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/Normalizer.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/Normalizer.proto

$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/NeuralNetwork.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/NeuralNetwork.proto

$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/ArrayFeatureExtractor.proto

$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/CategoricalMapping.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/CategoricalMapping.proto

$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/DictVectorizer.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/DictVectorizer.proto

$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/FeatureVectorizer.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/FeatureVectorizer.proto

$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/GLMRegressor.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/GLMRegressor.proto

$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/GLMClassifier.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/GLMClassifier.proto

$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/Identity.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/Identity.proto

$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/Imputer.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/Imputer.proto

$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc\
$(BUILD_ROOT)/contrib/libs/coreml/Model.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/coreml/Model.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/coreml/Model.proto

$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/coreml'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/containers/dense_hash/liblibrary-containers-dense_hash.a\
$(BUILD_ROOT)/library/containers/dense_hash/liblibrary-containers-dense_hash.a.mf\
        ::\
        $(BUILD_ROOT)/library/containers/dense_hash/dense_hash.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/containers/dense_hash'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-containers-dense_hash -o library/containers/dense_hash/liblibrary-containers-dense_hash.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/containers/dense_hash/liblibrary-containers-dense_hash.a' '$(BUILD_ROOT)/library/containers/dense_hash/dense_hash.cpp.o'

$(BUILD_ROOT)/library/containers/dense_hash/dense_hash.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/containers/dense_hash/dense_hash.cpp\

	mkdir -p '$(BUILD_ROOT)/library/containers/dense_hash'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/containers/dense_hash/dense_hash.cpp.o' '$(SOURCE_ROOT)/library/containers/dense_hash/dense_hash.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a\
$(BUILD_ROOT)/contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/src/util.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-flatbuffers -o contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a.mf -t LIBRARY -Ya,lics APACHE2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a' '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/util.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp.o'

$(BUILD_ROOT)/contrib/libs/flatbuffers/src/util.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/util.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/util.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/util.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/reflection.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_text.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_parser.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a\
$(BUILD_ROOT)/catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/__/__/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model/flatbuffers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-model-flatbuffers -o catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/model/flatbuffers/liblibs-model-flatbuffers.a' '$(BUILD_ROOT)/catboost/libs/model/flatbuffers/__/__/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/catboost/libs/model/flatbuffers/__/__/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model/flatbuffers/__/__/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/flatbuffers/__/__/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/libcatboost-libs-model.a\
$(BUILD_ROOT)/catboost/libs/model/libcatboost-libs-model.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/model_build_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/formula_evaluator.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/static_ctr_provider.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/online_ctr.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/model.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/features.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/ctr_value_table.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/ctr_provider.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/ctr_data.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/coreml_helpers.cpp.o\
        $(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-model -o catboost/libs/model/libcatboost-libs-model.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/model/libcatboost-libs-model.a' '$(BUILD_ROOT)/catboost/libs/model/model_build_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/formula_evaluator.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/static_ctr_provider.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/online_ctr.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/model.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/features.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/ctr_value_table.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/ctr_provider.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/ctr_data.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/coreml_helpers.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/model/model_build_helper.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/model_build_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/model_build_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/model_build_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/flatc/flatc\
        $(SOURCE_ROOT)/catboost/libs/model/flatbuffers/model.fbs\
        $(SOURCE_ROOT)/build/scripts/stdout2stderr.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model/flatbuffers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/plugins/_unpickler.py' --src-root '$(SOURCE_ROOT)' --build-root '$(BUILD_ROOT)' --data gAJjZmxhdGMKRmxhdGMKcQApgXEBfXECKFUKX2luY2xfZGlyc3EDXXEEKFUCJFNxBVUCJEJxBmVVBV9wYXRocQdVLCRTL2NhdGJvb3N0L2xpYnMvbW9kZWwvZmxhdGJ1ZmZlcnMvbW9kZWwuZmJzcQh1hXEJYi4= --tools 1 '$(BUILD_ROOT)/contrib/tools/flatc/flatc'

$(BUILD_ROOT)/contrib/tools/flatc/flatc\
$(BUILD_ROOT)/contrib/tools/flatc/flatc.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a\
        $(BUILD_ROOT)/contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a\
        $(BUILD_ROOT)/contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a\
        $(BUILD_ROOT)/library/malloc/api/liblibrary-malloc-api.a\
        $(BUILD_ROOT)/library/lfalloc/liblibrary-lfalloc.a\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a\
        $(BUILD_ROOT)/contrib/tools/flatc/__/__/libs/flatbuffers/src/flatc_main.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_exe.py\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/flatc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name flatc -o contrib/tools/flatc/flatc.mf -t PROGRAM -Ya,lics -Ya,peers contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a
	cd $(BUILD_ROOT) && '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_exe.py' '$(CXX)' '$(BUILD_ROOT)/contrib/tools/flatc/__/__/libs/flatbuffers/src/flatc_main.cpp.o' -o '$(BUILD_ROOT)/contrib/tools/flatc/flatc' -rdynamic -Wl,--start-group contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a contrib/libs/cxxsupp/libcontrib-libs-cxxsupp.a library/lfalloc/liblibrary-lfalloc.a contrib/libs/flatbuffers/libcontrib-libs-flatbuffers.a contrib/libs/cxxsupp/libcxx/liblibs-cxxsupp-libcxx.a library/malloc/api/liblibrary-malloc-api.a contrib/libs/cxxsupp/libcxxrt/liblibs-cxxsupp-libcxxrt.a contrib/libs/cppdemangle/libcontrib-libs-cppdemangle.a contrib/libs/libunwind_master/libcontrib-libs-libunwind_master.a contrib/libs/cxxsupp/builtins/liblibs-cxxsupp-builtins.a -Wl,--end-group -ldl -lrt -Wl,--no-as-needed -lpthread -nodefaultlibs -lpthread -lc -lm

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a\
$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_python.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_php.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_js.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_general.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_fbs.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_cpp.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/flatc.cpp.o\
        $(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/code_generators.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name libs-flatbuffers-flatc -o contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a.mf -t LIBRARY -Ya,lics APACHE2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/liblibs-flatbuffers-flatc.a' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_python.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_php.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_js.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_general.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_fbs.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_cpp.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/flatc.cpp.o' '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/code_generators.cpp.o'

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_python.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_python.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_python.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_python.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_php.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_php.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_php.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_php.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_js.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_js.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_js.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_js.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_general.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_general.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_general.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_general.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_fbs.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_fbs.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_fbs.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_fbs.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_cpp.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_cpp.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/idl_gen_cpp.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/idl_gen_cpp.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/flatc.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/flatc.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/flatc.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/flatc.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/code_generators.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/code_generators.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/flatbuffers/flatc/__/src/code_generators.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/code_generators.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/tools/flatc/__/__/libs/flatbuffers/src/flatc_main.cpp.o\
        ::\
        $(SOURCE_ROOT)/contrib/libs/flatbuffers/src/flatc_main.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/tools/flatc/__/__/libs/flatbuffers/src'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/tools/flatc/__/__/libs/flatbuffers/src/flatc_main.cpp.o' '$(SOURCE_ROOT)/contrib/libs/flatbuffers/src/flatc_main.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/flatbuffers/include' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/flatc/flatc\
        $(SOURCE_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs\
        $(SOURCE_ROOT)/build/scripts/stdout2stderr.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model/flatbuffers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/plugins/_unpickler.py' --src-root '$(SOURCE_ROOT)' --build-root '$(BUILD_ROOT)' --data gAJjZmxhdGMKRmxhdGMKcQApgXEBfXECKFUKX2luY2xfZGlyc3EDXXEEKFUCJFNxBVUCJEJxBmVVBV9wYXRocQdVLyRTL2NhdGJvb3N0L2xpYnMvbW9kZWwvZmxhdGJ1ZmZlcnMvY3RyX2RhdGEuZmJzcQh1hXEJYi4= --tools 1 '$(BUILD_ROOT)/contrib/tools/flatc/flatc'

$(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/flatc/flatc\
        $(SOURCE_ROOT)/catboost/libs/model/flatbuffers/features.fbs\
        $(SOURCE_ROOT)/build/scripts/stdout2stderr.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model/flatbuffers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/plugins/_unpickler.py' --src-root '$(SOURCE_ROOT)' --build-root '$(BUILD_ROOT)' --data gAJjZmxhdGMKRmxhdGMKcQApgXEBfXECKFUKX2luY2xfZGlyc3EDXXEEKFUCJFNxBVUCJEJxBmVVBV9wYXRocQdVLyRTL2NhdGJvb3N0L2xpYnMvbW9kZWwvZmxhdGJ1ZmZlcnMvZmVhdHVyZXMuZmJzcQh1hXEJYi4= --tools 1 '$(BUILD_ROOT)/contrib/tools/flatc/flatc'

$(BUILD_ROOT)/catboost/libs/model/formula_evaluator.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/formula_evaluator.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/formula_evaluator.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/formula_evaluator.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/static_ctr_provider.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/static_ctr_provider.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/static_ctr_provider.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/static_ctr_provider.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/online_ctr.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/online_ctr.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/online_ctr.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/online_ctr.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/model.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/model.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/model.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/model.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/features.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/features.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/features.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/features.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/ctr_value_table.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/ctr_value_table.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/ctr_value_table.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/ctr_value_table.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/ctr_provider.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/ctr_provider.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/ctr_provider.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/ctr_provider.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/ctr_data.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/ctr_data.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/ctr_data.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/ctr_data.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/coreml_helpers.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureTypes.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DataStructures.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/TreeEnsemble.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Scaler.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/SVM.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/OneHotEncoder.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Normalizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/NeuralNetwork.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/ArrayFeatureExtractor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/CategoricalMapping.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/DictVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/FeatureVectorizer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMRegressor.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/GLMClassifier.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Identity.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Imputer.pb.h\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.cc\
        $(BUILD_ROOT)/contrib/libs/coreml/Model.pb.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/coreml_helpers.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/coreml_helpers.cpp.o' '$(SOURCE_ROOT)/catboost/libs/model/coreml_helpers.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/model/split.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/model'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/model/split.h' --include-path catboost/libs/model/split.h --output '$(BUILD_ROOT)/catboost/libs/model/split.h_serialized.cpp'

$(BUILD_ROOT)/catboost/libs/data/libcatboost-libs-data.a\
$(BUILD_ROOT)/catboost/libs/data/libcatboost-libs-data.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/data/load_data.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/data'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-data -o catboost/libs/data/libcatboost-libs-data.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/data/libcatboost-libs-data.a' '$(BUILD_ROOT)/catboost/libs/data/load_data.cpp.o'

$(BUILD_ROOT)/catboost/libs/data/load_data.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/data/load_data.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/data'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/data/load_data.cpp.o' '$(SOURCE_ROOT)/catboost/libs/data/load_data.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/libcontrib-libs-tensorboard.a\
$(BUILD_ROOT)/contrib/libs/tensorboard/libcontrib-libs-tensorboard.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc.o\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-tensorboard -o contrib/libs/tensorboard/libcontrib-libs-tensorboard.a.mf -t LIBRARY -Ya,lics APACHE2 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/tensorboard/libcontrib-libs-tensorboard.a' '$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc.o'

$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/types.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/types.proto

$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/tensor_shape.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/tensor_shape.proto

$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/resource_handle.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/resource_handle.proto

$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/tensor.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/tensor.proto

$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/summary.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/summary.proto

$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc.o' '$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
$(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\
        ::\
        $(BUILD_ROOT)/contrib/tools/protoc/protoc\
        $(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide\
        $(SOURCE_ROOT)/contrib/libs/tensorboard/event.proto\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/tensorboard'
	cd $(SOURCE_ROOT) && '$(BUILD_ROOT)/contrib/tools/protoc/protoc' -I=./ '-I=$(SOURCE_ROOT)/' '-I=$(BUILD_ROOT)' '-I=$(SOURCE_ROOT)/contrib/libs/protobuf' '--cpp_out=$(BUILD_ROOT)/' '--cpp_styleguide_out=$(BUILD_ROOT)/' '--plugin=protoc-gen-cpp_styleguide=$(BUILD_ROOT)/contrib/tools/protoc/plugins/cpp_styleguide/cpp_styleguide' contrib/libs/tensorboard/event.proto

$(BUILD_ROOT)/catboost/libs/loggers/libcatboost-libs-loggers.a\
$(BUILD_ROOT)/catboost/libs/loggers/libcatboost-libs-loggers.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/loggers/logger.cpp.o\
        $(BUILD_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/loggers'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-loggers -o catboost/libs/loggers/libcatboost-libs-loggers.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/loggers/libcatboost-libs-loggers.a' '$(BUILD_ROOT)/catboost/libs/loggers/logger.cpp.o' '$(BUILD_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp.o'

$(BUILD_ROOT)/catboost/libs/loggers/logger.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\
        $(SOURCE_ROOT)/catboost/libs/loggers/logger.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/loggers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/loggers/logger.cpp.o' '$(SOURCE_ROOT)/catboost/libs/loggers/logger.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp.o\
        ::\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\
        $(SOURCE_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/loggers'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp.o' '$(SOURCE_ROOT)/catboost/libs/loggers/tensorboard_logger.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/libcatboost-libs-metrics.a\
$(BUILD_ROOT)/catboost/libs/metrics/libcatboost-libs-metrics.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/metrics/pfound.cpp.o\
        $(BUILD_ROOT)/catboost/libs/metrics/metric.cpp.o\
        $(BUILD_ROOT)/catboost/libs/metrics/auc.cpp.o\
        $(BUILD_ROOT)/catboost/libs/metrics/sample.cpp.o\
        $(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-metrics -o catboost/libs/metrics/libcatboost-libs-metrics.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/metrics/libcatboost-libs-metrics.a' '$(BUILD_ROOT)/catboost/libs/metrics/pfound.cpp.o' '$(BUILD_ROOT)/catboost/libs/metrics/metric.cpp.o' '$(BUILD_ROOT)/catboost/libs/metrics/auc.cpp.o' '$(BUILD_ROOT)/catboost/libs/metrics/sample.cpp.o' '$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/metrics/pfound.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/metrics/pfound.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/metrics/pfound.cpp.o' '$(SOURCE_ROOT)/catboost/libs/metrics/pfound.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/metric.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/metrics/metric.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/metrics/metric.cpp.o' '$(SOURCE_ROOT)/catboost/libs/metrics/metric.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/auc.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/metrics/auc.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/metrics/auc.cpp.o' '$(SOURCE_ROOT)/catboost/libs/metrics/auc.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/sample.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/metrics/sample.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/metrics/sample.cpp.o' '$(SOURCE_ROOT)/catboost/libs/metrics/sample.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/metrics/metric.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/metrics'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/metrics/metric.h' --include-path catboost/libs/metrics/metric.h --output '$(BUILD_ROOT)/catboost/libs/metrics/metric.h_serialized.cpp'

$(BUILD_ROOT)/library/statistics/liblibrary-statistics.a\
$(BUILD_ROOT)/library/statistics/liblibrary-statistics.a.mf\
        ::\
        $(BUILD_ROOT)/library/statistics/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/statistics'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-statistics -o library/statistics/liblibrary-statistics.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/statistics/liblibrary-statistics.a' '$(BUILD_ROOT)/library/statistics/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/library/statistics/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/library/statistics/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/statistics/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a\
$(BUILD_ROOT)/catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp.o\
        $(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/overfitting_detector'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-overfitting_detector -o catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/overfitting_detector/libcatboost-libs-overfitting_detector.a' '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp.o' '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp.o'

$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/overfitting_detector'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp.o' '$(SOURCE_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/overfitting_detector'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp.o' '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp\
        ::\
        $(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser\
        $(SOURCE_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/overfitting_detector'
	'$(BUILD_ROOT)/tools/enum_parser/enum_parser/enum_parser' '$(SOURCE_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h' --include-path catboost/libs/overfitting_detector/overfitting_detector.h --output '$(BUILD_ROOT)/catboost/libs/overfitting_detector/overfitting_detector.h_serialized.cpp'

$(BUILD_ROOT)/library/dot_product/liblibrary-dot_product.a\
$(BUILD_ROOT)/library/dot_product/liblibrary-dot_product.a.mf\
        ::\
        $(BUILD_ROOT)/library/dot_product/dot_product.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/dot_product'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-dot_product -o library/dot_product/liblibrary-dot_product.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/dot_product/liblibrary-dot_product.a' '$(BUILD_ROOT)/library/dot_product/dot_product.cpp.o'

$(BUILD_ROOT)/library/dot_product/dot_product.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/dot_product/dot_product.cpp\

	mkdir -p '$(BUILD_ROOT)/library/dot_product'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/dot_product/dot_product.cpp.o' '$(SOURCE_ROOT)/library/dot_product/dot_product.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/contrib/libs/fmath/libcontrib-libs-fmath.a\
$(BUILD_ROOT)/contrib/libs/fmath/libcontrib-libs-fmath.a.mf\
        ::\
        $(BUILD_ROOT)/contrib/libs/fmath/__/__/__/build/scripts/_fake_src.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/fmath'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name contrib-libs-fmath -o contrib/libs/fmath/libcontrib-libs-fmath.a.mf -t LIBRARY -Ya,lics BSD3 -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/contrib/libs/fmath/libcontrib-libs-fmath.a' '$(BUILD_ROOT)/contrib/libs/fmath/__/__/__/build/scripts/_fake_src.cpp.o'

$(BUILD_ROOT)/contrib/libs/fmath/__/__/__/build/scripts/_fake_src.cpp.o\
        ::\
        $(SOURCE_ROOT)/build/scripts/_fake_src.cpp\

	mkdir -p '$(BUILD_ROOT)/contrib/libs/fmath/__/__/__/build/scripts'
	'$(CXX)' -c -o '$(BUILD_ROOT)/contrib/libs/fmath/__/__/__/build/scripts/_fake_src.cpp.o' '$(SOURCE_ROOT)/build/scripts/_fake_src.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/fast_exp/liblibrary-fast_exp.a\
$(BUILD_ROOT)/library/fast_exp/liblibrary-fast_exp.a.mf\
        ::\
        $(BUILD_ROOT)/library/fast_exp/fast_exp.cpp.o\
        $(BUILD_ROOT)/library/fast_exp/fast_exp_sse2.cpp.o\
        $(BUILD_ROOT)/library/fast_exp/fast_exp_avx2.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/fast_exp'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-fast_exp -o library/fast_exp/liblibrary-fast_exp.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/fast_exp/liblibrary-fast_exp.a' '$(BUILD_ROOT)/library/fast_exp/fast_exp.cpp.o' '$(BUILD_ROOT)/library/fast_exp/fast_exp_sse2.cpp.o' '$(BUILD_ROOT)/library/fast_exp/fast_exp_avx2.cpp.o'

$(BUILD_ROOT)/library/fast_exp/fast_exp.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/fast_exp/fast_exp.cpp\

	mkdir -p '$(BUILD_ROOT)/library/fast_exp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/fast_exp/fast_exp.cpp.o' '$(SOURCE_ROOT)/library/fast_exp/fast_exp.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/fast_exp/fast_exp_sse2.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/fast_exp/fast_exp_sse2.cpp\

	mkdir -p '$(BUILD_ROOT)/library/fast_exp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/fast_exp/fast_exp_sse2.cpp.o' '$(SOURCE_ROOT)/library/fast_exp/fast_exp_sse2.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++ -msse2

$(BUILD_ROOT)/library/fast_exp/fast_exp_avx2.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/fast_exp/fast_exp_avx2.cpp\

	mkdir -p '$(BUILD_ROOT)/library/fast_exp'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/fast_exp/fast_exp_avx2.cpp.o' '$(SOURCE_ROOT)/library/fast_exp/fast_exp_avx2.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++ -mavx2

$(BUILD_ROOT)/library/fast_log/liblibrary-fast_log.a\
$(BUILD_ROOT)/library/fast_log/liblibrary-fast_log.a.mf\
        ::\
        $(BUILD_ROOT)/library/fast_log/fast_log.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/fast_log'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-fast_log -o library/fast_log/liblibrary-fast_log.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/fast_log/liblibrary-fast_log.a' '$(BUILD_ROOT)/library/fast_log/fast_log.cpp.o'

$(BUILD_ROOT)/library/fast_log/fast_log.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/fast_log/fast_log.cpp\

	mkdir -p '$(BUILD_ROOT)/library/fast_log'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/fast_log/fast_log.cpp.o' '$(SOURCE_ROOT)/library/fast_log/fast_log.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/object_factory/liblibrary-object_factory.a\
$(BUILD_ROOT)/library/object_factory/liblibrary-object_factory.a.mf\
        ::\
        $(BUILD_ROOT)/library/object_factory/object_factory.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/object_factory'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-object_factory -o library/object_factory/liblibrary-object_factory.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/object_factory/liblibrary-object_factory.a' '$(BUILD_ROOT)/library/object_factory/object_factory.cpp.o'

$(BUILD_ROOT)/library/object_factory/object_factory.cpp.o\
        ::\
        $(SOURCE_ROOT)/library/object_factory/object_factory.cpp\

	mkdir -p '$(BUILD_ROOT)/library/object_factory'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/object_factory/object_factory.cpp.o' '$(SOURCE_ROOT)/library/object_factory/object_factory.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/libcatboost-libs-algo.a\
$(BUILD_ROOT)/catboost/libs/algo/libcatboost-libs-algo.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/algo/calc_score_cache.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/cv_data_partition.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/helpers.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/tree_print.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_map.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/train.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/target_classifier.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/split.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/score_calcer.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/ctr_helper.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/online_predictor.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/online_ctr.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/learn_context.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/index_hash_calcer.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/index_calcer.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/full_features.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/fold.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/features_layout.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/error_functions.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/apply.cpp.o\
        $(BUILD_ROOT)/catboost/libs/algo/plot.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-algo -o catboost/libs/algo/libcatboost-libs-algo.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/algo/libcatboost-libs-algo.a' '$(BUILD_ROOT)/catboost/libs/algo/calc_score_cache.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/cv_data_partition.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/helpers.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/tree_print.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_map.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/train.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/target_classifier.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/split.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/score_calcer.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/ctr_helper.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/online_predictor.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/online_ctr.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/learn_context.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/index_hash_calcer.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/index_calcer.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/full_features.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/fold.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/features_layout.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/error_functions.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/apply.cpp.o' '$(BUILD_ROOT)/catboost/libs/algo/plot.cpp.o'

$(BUILD_ROOT)/catboost/libs/algo/calc_score_cache.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/calc_score_cache.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/calc_score_cache.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/calc_score_cache.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/cv_data_partition.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/cv_data_partition.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/cv_data_partition.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/cv_data_partition.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/helpers.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/helpers.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/helpers.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/helpers.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/tree_print.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/tree_print.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/tree_print.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/tree_print.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_user_querywise.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_user_per_object.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_rmse.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_query_soft_max.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_query_rmse.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_quantile.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_poisson.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_pair_logit.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_multi_class_one_vs_all.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_multi_class.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_map.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_map.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_map.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_map.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_logloss.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_log_lin_quantile.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_custom.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train_one_iter_cross_entropy.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/train.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/train.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/train.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/train.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/target_classifier.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/algo/target_classifier.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/target_classifier.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/target_classifier.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/split.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/split.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/split.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/split.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/score_calcer.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/score_calcer.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/score_calcer.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/score_calcer.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/ctr_helper.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/ctr_helper.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/ctr_helper.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/ctr_helper.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/online_predictor.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/algo/online_predictor.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/online_predictor.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/online_predictor.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/online_ctr.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/online_ctr.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/online_ctr.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/online_ctr.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/learn_context.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/learn_context.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/learn_context.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/learn_context.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/index_hash_calcer.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/index_hash_calcer.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/index_hash_calcer.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/index_hash_calcer.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/index_calcer.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/index_calcer.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/index_calcer.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/index_calcer.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/greedy_tensor_search.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/full_features.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/full_features.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/full_features.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/full_features.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/fold.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/fold.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/fold.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/fold.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/features_layout.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/algo/features_layout.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/features_layout.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/features_layout.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/error_functions.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/algo/error_functions.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/error_functions.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/error_functions.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/apply.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/apply.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/apply.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/apply.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/algo/plot.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/algo/plot.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/algo'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/algo/plot.cpp.o' '$(SOURCE_ROOT)/catboost/libs/algo/plot.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/fstr/libcatboost-libs-fstr.a\
$(BUILD_ROOT)/catboost/libs/fstr/libcatboost-libs-fstr.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/fstr/calc_fstr.cpp.o\
        $(BUILD_ROOT)/catboost/libs/fstr/doc_fstr.cpp.o\
        $(BUILD_ROOT)/catboost/libs/fstr/feature_str.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/fstr'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-fstr -o catboost/libs/fstr/libcatboost-libs-fstr.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/fstr/libcatboost-libs-fstr.a' '$(BUILD_ROOT)/catboost/libs/fstr/calc_fstr.cpp.o' '$(BUILD_ROOT)/catboost/libs/fstr/doc_fstr.cpp.o' '$(BUILD_ROOT)/catboost/libs/fstr/feature_str.cpp.o'

$(BUILD_ROOT)/catboost/libs/fstr/calc_fstr.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/fstr/calc_fstr.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/fstr'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/fstr/calc_fstr.cpp.o' '$(SOURCE_ROOT)/catboost/libs/fstr/calc_fstr.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/fstr/doc_fstr.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/libs/fstr/doc_fstr.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/fstr'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/fstr/doc_fstr.cpp.o' '$(SOURCE_ROOT)/catboost/libs/fstr/doc_fstr.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/fstr/feature_str.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/libs/fstr/feature_str.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/fstr'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/fstr/feature_str.cpp.o' '$(SOURCE_ROOT)/catboost/libs/fstr/feature_str.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/train_lib/train_model.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\
        $(SOURCE_ROOT)/catboost/libs/train_lib/train_model.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/train_lib'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/train_lib/train_model.cpp.o' '$(SOURCE_ROOT)/catboost/libs/train_lib/train_model.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/libs/train_lib/libcatboost-libs-train_lib.a\
$(BUILD_ROOT)/catboost/libs/train_lib/libcatboost-libs-train_lib.a.mf\
        ::\
        $(BUILD_ROOT)/catboost/libs/train_lib/cross_validation.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/train_lib'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name catboost-libs-train_lib -o catboost/libs/train_lib/libcatboost-libs-train_lib.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/catboost/libs/train_lib/libcatboost-libs-train_lib.a' '$(BUILD_ROOT)/catboost/libs/train_lib/cross_validation.cpp.o'

$(BUILD_ROOT)/catboost/libs/train_lib/cross_validation.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/types.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor_shape.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/resource_handle.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/tensor.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/summary.pb.h\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.cc\
        $(BUILD_ROOT)/contrib/libs/tensorboard/event.pb.h\
        $(SOURCE_ROOT)/catboost/libs/train_lib/cross_validation.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/libs/train_lib'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/libs/train_lib/cross_validation.cpp.o' '$(SOURCE_ROOT)/catboost/libs/train_lib/cross_validation.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/svnversion/liblibrary-svnversion.a\
$(BUILD_ROOT)/library/svnversion/liblibrary-svnversion.a.mf\
        ::\
        $(BUILD_ROOT)/library/svnversion/svnversion.cpp.o\
        $(SOURCE_ROOT)/build/scripts/generate_mf.py\
        $(SOURCE_ROOT)/build/scripts/link_lib.py\

	mkdir -p '$(BUILD_ROOT)/library/svnversion'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/generate_mf.py' --build-root '$(BUILD_ROOT)' --module-name library-svnversion -o library/svnversion/liblibrary-svnversion.a.mf -t LIBRARY -Ya,lics -Ya,peers
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/link_lib.py' ar AR '$(BUILD_ROOT)' None '$(BUILD_ROOT)/library/svnversion/liblibrary-svnversion.a' '$(BUILD_ROOT)/library/svnversion/svnversion.cpp.o'

$(BUILD_ROOT)/library/svnversion/svnversion.cpp.o\
        ::\
        $(BUILD_ROOT)/library/svnversion/svnversion_data.h\
        $(SOURCE_ROOT)/library/svnversion/svnversion.cpp\

	mkdir -p '$(BUILD_ROOT)/library/svnversion'
	'$(CXX)' -c -o '$(BUILD_ROOT)/library/svnversion/svnversion.cpp.o' '$(SOURCE_ROOT)/library/svnversion/svnversion.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/library/svnversion/svnversion_data.h\
        ::\
        $(SOURCE_ROOT)/build/scripts/yield_line.py\
        $(SOURCE_ROOT)/build/scripts/xargs.py\
        $(SOURCE_ROOT)/build/scripts/svn_version_gen.py\
        $(SOURCE_ROOT)/.svn/wc.db\

	mkdir -p '$(BUILD_ROOT)/library/svnversion'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/yield_line.py' -- '$(BUILD_ROOT)/library/svnversion/__args' '$(SOURCE_ROOT)'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/yield_line.py' -- '$(BUILD_ROOT)/library/svnversion/__args' '$(BUILD_ROOT)'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/yield_line.py' -- '$(BUILD_ROOT)/library/svnversion/__args' '$(PYTHON)/python'
	'$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/xargs.py' -- '$(BUILD_ROOT)/library/svnversion/__args' '$(PYTHON)/python' '$(SOURCE_ROOT)/build/scripts/svn_version_gen.py' '$(BUILD_ROOT)/library/svnversion/svnversion_data.h'

$(BUILD_ROOT)/catboost/app/cmd_line.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/app/cmd_line.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/cmd_line.cpp.o' '$(SOURCE_ROOT)/catboost/app/cmd_line.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/bind_options.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/app/bind_options.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/bind_options.cpp.o' '$(SOURCE_ROOT)/catboost/app/bind_options.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/mode_plot.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/app/mode_plot.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/mode_plot.cpp.o' '$(SOURCE_ROOT)/catboost/app/mode_plot.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/mode_fstr.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/app/mode_fstr.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/mode_fstr.cpp.o' '$(SOURCE_ROOT)/catboost/app/mode_fstr.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/mode_fit.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/app/mode_fit.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/mode_fit.cpp.o' '$(SOURCE_ROOT)/catboost/app/mode_fit.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/mode_calc.cpp.o\
        ::\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/model.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/ctr_data.fbs.h\
        $(BUILD_ROOT)/catboost/libs/model/flatbuffers/features.fbs.h\
        $(SOURCE_ROOT)/catboost/app/mode_calc.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/mode_calc.cpp.o' '$(SOURCE_ROOT)/catboost/app/mode_calc.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++

$(BUILD_ROOT)/catboost/app/main.cpp.o\
        ::\
        $(SOURCE_ROOT)/catboost/app/main.cpp\

	mkdir -p '$(BUILD_ROOT)/catboost/app'
	'$(CXX)' -c -o '$(BUILD_ROOT)/catboost/app/main.cpp.o' '$(SOURCE_ROOT)/catboost/app/main.cpp' '-I$(BUILD_ROOT)' '-I$(SOURCE_ROOT)' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxxrt' '-I$(SOURCE_ROOT)/contrib/libs/cxxsupp/libcxx/include' '-I$(SOURCE_ROOT)/contrib/libs/protobuf' '-I$(SOURCE_ROOT)/contrib/libs/protobuf/google/protobuf' -Woverloaded-virtual -Wno-invalid-offsetof -Wno-attributes -Wno-undefined-var-template -std=c++14 -pipe -m64 -msse -msse3 -msse2 -fstack-protector -Wno-inconsistent-missing-override -g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -DGNU -D_GNU_SOURCE -DSSE_ENABLED=1 -DSSE3_ENABLED=1 -DSSE2_ENABLED=1 -UNDEBUG -D_THREAD_SAFE -D_PTHREADS -D_REENTRANT -D__LONG_LONG_SUPPORTED -Wall -W -Wno-parentheses -Wno-deprecated -nostdinc++ -DFAKEID=r3367582 '-DARCADIA_ROOT=$(SOURCE_ROOT)' '-DARCADIA_BUILD_ROOT=$(BUILD_ROOT)' -nostdinc++
