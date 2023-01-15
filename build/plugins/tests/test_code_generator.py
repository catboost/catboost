from build.plugins import code_generator


def test_include_parser():
    template_file = """
        @ from 'util/namespace.macro' import namespace, change_namespace, close_namespaces
        @ import 'market/tools/code_generator/templates/serialization/json.macro' as json
        @ import 'market/tools/code_generator/templates/serialization/request_parameters.macro' as rp
        #include <sss/abcdefg.h>
        #include<fff/asd>
        #include "hhh/quququ.h"
        """

    includes, induced = code_generator.CodeGeneratorTemplateParser.parse_includes(template_file.split('\n'))
    assert includes == ['util/namespace.macro', 'market/tools/code_generator/templates/serialization/json.macro', 'market/tools/code_generator/templates/serialization/request_parameters.macro']
    assert induced == ['sss/abcdefg.h', 'fff/asd', 'hhh/quququ.h']
