_JAVA_LIBRARY()



SET(PACKAGE_PREFIX com.google.protobuf)

PEERDIR(
    contrib/java/com/google/code/gson/gson/2.8.0
    contrib/java/com/google/guava/guava/21.0
)

SRCDIR(contrib/libs/protobuf)

SRCS(
    util/TimeUtil.java
    util/FieldMaskTree.java
    util/Durations.java
    util/Timestamps.java
    util/JsonFormat.java
    util/FieldMaskUtil.java
    core/BlockingService.java
    core/WireFormat.java
    core/InvalidProtocolBufferException.java
    core/ExtensionLite.java
    core/AbstractMessage.java
    core/LazyStringArrayList.java
    core/Descriptors.java
    core/LazyField.java
    core/Extension.java
    core/LongArrayList.java
    core/NioByteString.java
    core/CodedOutputStream.java
    core/MessageLiteToString.java
    core/FieldSet.java
    core/RepeatedFieldBuilderV3.java
    core/UninitializedMessageException.java
    core/RepeatedFieldBuilder.java
    core/Parser.java
    core/UnsafeUtil.java
    core/Utf8.java
    core/RpcChannel.java
    core/LazyStringList.java
    core/ByteBufferWriter.java
    core/ProtocolStringList.java
    core/RpcController.java
    core/ExtensionRegistryLite.java
    core/UnsafeByteOperations.java
    core/ProtobufArrayList.java
    core/Internal.java
    core/MapFieldLite.java
    core/SingleFieldBuilder.java
    core/UnmodifiableLazyStringList.java
    core/SmallSortedMap.java
    core/MessageReflection.java
    core/SingleFieldBuilderV3.java
    core/DynamicMessage.java
    core/TextFormatParseInfoTree.java
    core/Service.java
    core/GeneratedMessage.java
    core/AbstractProtobufList.java
    core/ProtocolMessageEnum.java
    core/GeneratedMessageLite.java
    core/UnknownFieldSetLite.java
    core/MessageOrBuilder.java
    core/UnknownFieldSet.java
    core/BooleanArrayList.java
    core/ExperimentalApi.java
    core/AbstractParser.java
    core/TextFormatEscaper.java
    core/ByteOutput.java
    core/MapField.java
    core/LazyFieldLite.java
    core/BlockingRpcChannel.java
    core/MapEntry.java
    core/IntArrayList.java
    core/ExtensionRegistryFactory.java
    core/MapEntryLite.java
    core/AbstractMessageLite.java
    core/ByteString.java
    core/MutabilityOracle.java
    core/CodedInputStream.java
    core/DoubleArrayList.java
    core/RopeByteString.java
    core/TextFormatParseLocation.java
    core/ServiceException.java
    core/GeneratedMessageV3.java
    core/TextFormat.java
    core/RpcUtil.java
    core/MessageLite.java
    core/FloatArrayList.java
    core/MessageLiteOrBuilder.java
    core/ExtensionRegistry.java
    core/Message.java
    core/RpcCallback.java

    google/protobuf/field_mask.proto
#    google/protobuf/type.proto
    google/protobuf/struct.proto
#    google/protobuf/api.proto
    google/protobuf/empty.proto
    google/protobuf/any.proto
    google/protobuf/duration.proto
    google/protobuf/source_context.proto
    google/protobuf/wrappers.proto
    google/protobuf/timestamp.proto
    google/protobuf/descriptor.proto
)

END()
