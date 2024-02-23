#!/usr/bin/env python

import os
import string


def generate_project(src_dir, substitution_dict, dst_dir):
    for path in ('pom.xml', 'macros/pom.xml', 'core/pom.xml'):
        with open(os.path.join(src_dir, path)) as src_file:
            src = src_file.read()

        dst = string.Template(src).safe_substitute(substitution_dict)

        dst_path = os.path.join(dst_dir, path)
        dst_path_dir = os.path.dirname(dst_path)
        if not os.path.exists(dst_path_dir):
            os.makedirs(dst_path_dir)

        with open(dst_path, 'w') as dst_file:
            dst_file.write(dst)


global_substitition_dict = {
    'catboost_version_placeholder': '1.2.3',
    'relative_global_project_root_placeholder': '../..'
}

scala_maven_plugin_configuration_scala_pre_2_13 = """
<compilerPlugins>
    <compilerPlugin>
        <groupId>org.scalamacros</groupId>
        <artifactId>paradise_${scala.version}</artifactId>
        <version>2.1.1</version>
    </compilerPlugin>
</compilerPlugins>
"""

scala_maven_plugin_configuration_scala_2_13 = """
<args>
    <arg>-Ymacro-annotations</arg>
</args>
"""

configs = [
    {
        'dst_dir' : '../projects/spark_2.3_2.11',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_2_x/scala',
            'scala_compat_version_placeholder': '2.11',
            'scala_version_placeholder': '2.11.12',
            'spark_compat_version_placeholder': '2.3',
            'spark_version_placeholder': '2.3.0',
            'hadoop_version_placeholder': '2.7.3',
            'json4s_version_placeholder': '3.2.11',
            'jackson_databind_version_placeholder': '2.6.7.5',
            'jackson_module_scala_version_placeholder': '2.6.7.1',
            'netty_all_version_placeholer': '4.1.42.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_2.4_2.11',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_2_x/scala',
            'scala_compat_version_placeholder': '2.11',
            'scala_version_placeholder': '2.11.12',
            'spark_compat_version_placeholder': '2.4',
            'spark_version_placeholder': '2.4.0',
            'hadoop_version_placeholder': '2.7.3',
            'json4s_version_placeholder': '3.5.3',
            'jackson_databind_version_placeholder': '2.6.7.5',
            'jackson_module_scala_version_placeholder': '2.6.7.1',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_2.4_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_2_x/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '2.4',
            'spark_version_placeholder': '2.4.1',
            'hadoop_version_placeholder': '2.7.3',
            'json4s_version_placeholder': '3.5.3',
            'jackson_databind_version_placeholder': '2.6.7.5',
            'jackson_module_scala_version_placeholder': '2.6.7.1',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.0_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.0',
            'spark_version_placeholder': '3.0.1',
            'hadoop_version_placeholder': '2.7.3',
            'json4s_version_placeholder': '3.6.6',
            'jackson_databind_version_placeholder': '2.10.5.1',
            'jackson_module_scala_version_placeholder': '2.10.5',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.1_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.1',
            'spark_version_placeholder': '3.1.0',
            'hadoop_version_placeholder': '3.2.0',
            'json4s_version_placeholder': '3.7.0-M5',
            'jackson_databind_version_placeholder': '2.10.5.1',
            'jackson_module_scala_version_placeholder': '2.10.5',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.2_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.2',
            'spark_version_placeholder': '3.2.0',
            'hadoop_version_placeholder': '3.3.1',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.12.7.1',
            'jackson_module_scala_version_placeholder': '2.12.7',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.2_2.13',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.13',
            'scala_version_placeholder': '2.13.5',
            'spark_compat_version_placeholder': '3.2',
            'spark_version_placeholder': '3.2.0',
            'hadoop_version_placeholder': '3.3.1',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.12.7.1',
            'jackson_module_scala_version_placeholder': '2.12.7',
            'netty_all_version_placeholer': '4.1.72.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.3_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.3',
            'spark_version_placeholder': '3.3.0',
            'hadoop_version_placeholder': '3.3.2',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.13.5',
            'jackson_module_scala_version_placeholder': '2.13.5',
            'netty_all_version_placeholer': '4.1.74.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.3_2.13',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.13',
            'scala_version_placeholder': '2.13.5',
            'spark_compat_version_placeholder': '3.3',
            'spark_version_placeholder': '3.3.0',
            'hadoop_version_placeholder': '3.3.2',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.13.5',
            'jackson_module_scala_version_placeholder': '2.13.5',
            'netty_all_version_placeholer': '4.1.74.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.4_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.4',
            'spark_version_placeholder': '3.4.0',
            'hadoop_version_placeholder': '3.3.4',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.14.2',
            'jackson_module_scala_version_placeholder': '2.14.2',
            'netty_all_version_placeholer': '4.1.87.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.4_2.13',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.0__3.4/scala',
            'scala_compat_version_placeholder': '2.13',
            'scala_version_placeholder': '2.13.5',
            'spark_compat_version_placeholder': '3.4',
            'spark_version_placeholder': '3.4.0',
            'hadoop_version_placeholder': '3.3.4',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.14.2',
            'jackson_module_scala_version_placeholder': '2.14.2',
            'netty_all_version_placeholer': '4.1.87.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.5_2.12',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.5/scala',
            'scala_compat_version_placeholder': '2.12',
            'scala_version_placeholder': '2.12.12',
            'spark_compat_version_placeholder': '3.5',
            'spark_version_placeholder': '3.5.0',
            'hadoop_version_placeholder': '3.3.4',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.15.2',
            'jackson_module_scala_version_placeholder': '2.15.2',
            'netty_all_version_placeholer': '4.1.96.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_pre_2_13
        }
    },
    {
        'dst_dir' : '../projects/spark_3.5_2.13',
        'substitution_dict' : {
            'version_specific_src_dir': 'spark_3.5/scala',
            'scala_compat_version_placeholder': '2.13',
            'scala_version_placeholder': '2.13.5',
            'spark_compat_version_placeholder': '3.5',
            'spark_version_placeholder': '3.5.0',
            'hadoop_version_placeholder': '3.3.4',
            'json4s_version_placeholder': '3.7.0-M11',
            'jackson_databind_version_placeholder': '2.15.2',
            'jackson_module_scala_version_placeholder': '2.15.2',
            'netty_all_version_placeholer': '4.1.96.Final',
            'scala_maven_plugin_configuration_placeholder': scala_maven_plugin_configuration_scala_2_13
        }
    }
]

current_dir = os.path.dirname(os.path.realpath(__file__))

for config in configs:
    substitution_dict = global_substitition_dict.copy()
    substitution_dict.update(config['substitution_dict'])
    generate_project(
        src_dir=current_dir,
        substitution_dict=substitution_dict,
        dst_dir=os.path.join(current_dir, config['dst_dir'])
    )
