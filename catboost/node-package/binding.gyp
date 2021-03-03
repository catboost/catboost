{
  'targets': [
    {
      'target_name': 'catboost-node',
      'sources': [ 'src/api_module.cpp' ],
      'include_dirs': [
                         "<!@(node -p \"require('node-addon-api').include\")",
                         "../libs/model_interface"
                      ],
      'dependencies': [ "<!(node -p \"require('node-addon-api').gyp\")" ],
      'libraries': [ 
                     "-L<!(echo \"$(pwd)/../libs/model_interface\")",
                     "-lcatboostmodel",
                     "-Wl,-rpath <!(echo \"$(pwd)/../libs/model_interface\")"
                   ],
      'cflags!': [ '-fno-exceptions' ],
      'cflags_cc!': [ '-fno-exceptions' ],
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        'MACOSX_DEPLOYMENT_TARGET': '10.7'
      },
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}
