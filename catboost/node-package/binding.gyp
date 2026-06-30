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
                         "<!@(node -p \"JSON.stringify(path.resolve(process.env['CATBOOST_SRC_PATH'] || path.join('..','..')))\")",
                      ],
      'dependencies': [ "<!(node -p \"require('node-addon-api').gyp\")" ],
      'conditions': [
        ['OS=="linux"', {
          'libraries': [
                     "-L<(module_root_dir)/build/catboost/libs/model_interface/",
                     "-lcatboostmodel",
                     "-Wl,-rpath <(module_root_dir)/build/catboost/libs/model_interface"
                   ],
        }],
        ['OS=="mac"', {
          'libraries': [
                     "-L<(module_root_dir)/build/catboost/libs/model_interface/",
                     "-lcatboostmodel",
                     "-Wl,-rpath,@loader_path/../catboost/libs/model_interface"
                   ],
          'postbuilds': [
            {
              'postbuild_name': 'Adjust load path',
              'action': [
                'install_name_tool',
                "-change",
                "libcatboostmodel.dylib.1",
                "@rpath/libcatboostmodel.dylib",
                "<(PRODUCT_DIR)/catboost-node.node"
              ]
            },
          ],
        }],
        ['OS=="win"', {
          'libraries': ["<(module_root_dir)/build/catboost/libs/model_interface/catboostmodel.lib"],
          'copies': [{
            'destination': '<(PRODUCT_DIR)',
            'files': [
              '<(module_root_dir)/build/catboost/libs/model_interface/catboostmodel.dll'
            ]
          }]
        }],
      ],
      'defines': [ 'NDEBUG' ],
      'cflags!': [ '-fno-exceptions' ],
      'cflags_cc!': [ '-fno-exceptions' ],
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        'MACOSX_DEPLOYMENT_TARGET': '11.0'
      },
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}
