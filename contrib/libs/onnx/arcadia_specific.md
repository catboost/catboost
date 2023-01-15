Changes for Arcadia contrib:

  * Some types have been replaced by Arcadia-specific types to be compatible with Arcadia-specific protobuf version in contrib
    std::string -> TString, std::vector -> TVector, std::ostringstream -> TStringStream, std::to_string -> ToString.
  * '-' changed to '_' in protobuf files' names, there were issues otherwise.
  * Protobuf definitions put into a separate 'proto' sub-library
  * ONNX_ML is always defined
  * Unused but big onnx/backend/test is removed
 
