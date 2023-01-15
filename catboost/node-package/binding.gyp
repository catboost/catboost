{
  'targets': [
    {
      'target_name': 'catboost-node',
      'sources': [ 
                      'src/api_helpers.cpp',
                      'src/api_module.cpp',
                      'src/model.cpp',
                 ],
      'include_dirs': [
                         "<!@(node -p \"require('node-addon-api').include\")",
                         "<!@(node -p \"process.env['CATBOOST_SRC_PATH'] || '../..'\")",
                         "<!@(node -p \"process.env['CATBOOST_SRC_PATH'] || '../..'\")/catboost/libs/model_interface",
                         "<!@(node -p \"process.env['CATBOOST_SRC_PATH'] || '../..'\")/contrib/libs/cxxsupp/system_stl/include",
                      ],
      'dependencies': [ "<!(node -p \"require('node-addon-api').gyp\")" ],
      'libraries': [ 
                     "-L<(module_root_dir)/build/catboost/libs/model_interface/",
                     "-lcatboostmodel",
                     "-Wl,-rpath <(module_root_dir)/build/catboost/libs/model_interface"
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
