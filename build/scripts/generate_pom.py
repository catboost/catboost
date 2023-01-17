import sys
import xml.etree.ElementTree as et
import argparse
import os
import json
import base64
import re


DEFAULT_YANDEX_GROUP_ID = 'ru.yandex'
DEFAULT_NAMESPACE = 'http://maven.apache.org/POM/4.0.0'
XSI_NAMESPACE = 'http://www.w3.org/2001/XMLSchema-instance'
SCHEMA_LOCATION = 'http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd'
MODEL_VERSION = '4.0.0'

MAVEN_PLUGIN_GROUP_ID = 'org.apache.maven.plugins'
MAVEN_PLUGIN_ARTIFACT_ID = 'maven-compiler-plugin'
MAVEN_PLUGIN_VERSION = '3.3'
JAVA_LANGUAGE_LEVEL = '1.8'

MAVEN_BUILD_HELPER_GROUP_ID = 'org.codehaus.mojo'
MAVEN_BUILD_HELPER_ARTIFACT_ID = 'build-helper-maven-plugin'
MAVEN_BUILD_HELPER_VERSION = '1.9.1'

MAVEN_EXEC_GROUP_ID = 'org.codehaus.mojo'
MAVEN_EXEC_ARTIFACT_ID = 'exec-maven-plugin'
MAVEN_EXEC_VERSION = '1.5.0'

MAVEN_SUREFIRE_GROUP_ID = 'org.apache.maven.plugins'
MAVEN_SUREFIRE_ARTIFACT_ID = 'maven-surefire-plugin'
MAVEN_SUREFIRE_VERSION = '2.12.2'


def target_from_contrib(target_path):
    return target_path.startswith('contrib')


def split_artifacts(s):
    m = re.match('^([^:]*:[^:]*:[^:]*:[^:]*)(.*)$', s)
    if not m or not m.groups():
        return []
    if not m.groups()[1].startswith('::'):
        return [m.groups()[0]]
    return [m.groups()[0]] + m.groups()[1].split('::')[1:]


def build_pom_and_export_to_maven(**kwargs):
    target_path = kwargs.get('target_path')
    target = kwargs.get('target')
    pom_path = kwargs.get('pom_path')
    source_dirs = kwargs.get('source_dirs')
    output_dir = kwargs.get('output_dir')
    final_name = kwargs.get('final_name')
    packaging = kwargs.get('packaging')
    target_dependencies = kwargs.get('target_dependencies')
    test_target_dependencies = kwargs.get('test_target_dependencies')
    test_target_dependencies_exclude = kwargs.get('test_target_dependencies_exclude')
    modules_path = kwargs.get('modules_path')
    prop_vars = kwargs.get('properties')
    external_jars = kwargs.get('external_jars')
    resources = kwargs.get('resources')
    run_java_programs = [json.loads(base64.b64decode(i)) for i in kwargs.get('run_java_programs')]
    test_source_dirs = kwargs.get('test_source_dirs')
    test_resource_dirs = kwargs.get('test_resource_dirs')

    modules = []

    def _indent(elem, level=0):
        ind = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = ind + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = ind
            for elem in elem:
                _indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = ind
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = ind

    project = et.Element(
        '{}{}{}project'.format('{', DEFAULT_NAMESPACE, '}'),
        attrib={'{}{}{}schemaLocation'.format('{', XSI_NAMESPACE, '}'): SCHEMA_LOCATION}
    )

    group_id, artifact_id, version = target.split(':')

    et.SubElement(project, 'modelVersion').text = MODEL_VERSION
    et.SubElement(project, 'groupId').text = group_id
    et.SubElement(project, 'artifactId').text = artifact_id
    et.SubElement(project, 'version').text = version
    et.SubElement(project, 'packaging').text = packaging

    properties = et.SubElement(project, 'properties')
    et.SubElement(properties, 'project.build.sourceEncoding').text = 'UTF-8'

    if prop_vars:
        for property, value in json.loads(base64.b64decode(prop_vars)).items():
            et.SubElement(properties, property).text = value

    if modules_path:
        with open(modules_path) as f:
            modules = [i.strip() for i in f if i.strip()]

    if modules:
        modules_el = et.SubElement(project, 'modules')
        for module in modules:
            et.SubElement(modules_el, 'module').text = module

    build = et.SubElement(project, 'build')
    if source_dirs:
        et.SubElement(build, 'sourceDirectory').text = source_dirs[0]
        source_dirs = source_dirs[1:]
    if test_source_dirs:
        et.SubElement(build, 'testSourceDirectory').text = test_source_dirs[0]
        test_source_dirs = test_source_dirs[1:]
    if output_dir:
        et.SubElement(build, 'outputDirectory').text = output_dir
    if final_name:
        et.SubElement(build, 'finalName').text = final_name
    if resources:
        resource_element = et.SubElement(et.SubElement(build, 'resources'), 'resource')
        et.SubElement(resource_element, 'directory').text = '${basedir}'
        includes = et.SubElement(resource_element, 'includes')
        for resource in resources:
            et.SubElement(includes, 'include').text = resource
    if test_resource_dirs:
        test_resource_element = et.SubElement(build, 'testResources')
        for test_resource_dir in test_resource_dirs:
            et.SubElement(et.SubElement(test_resource_element, 'testResource'), 'directory').text = '${basedir}' + (('/' + test_resource_dir) if test_resource_dir != '.' else '')

    plugins = et.SubElement(build, 'plugins')

    if packaging != 'pom':
        maven_plugin = et.SubElement(plugins, 'plugin')
        et.SubElement(maven_plugin, 'groupId').text = MAVEN_PLUGIN_GROUP_ID
        et.SubElement(maven_plugin, 'artifactId').text = MAVEN_PLUGIN_ARTIFACT_ID
        et.SubElement(maven_plugin, 'version').text = MAVEN_PLUGIN_VERSION
        configuration = et.SubElement(maven_plugin, 'configuration')
        et.SubElement(configuration, 'source').text = JAVA_LANGUAGE_LEVEL
        et.SubElement(configuration, 'target').text = JAVA_LANGUAGE_LEVEL

    if source_dirs or external_jars or test_source_dirs:
        build_helper_plugin = et.SubElement(plugins, 'plugin')
        et.SubElement(build_helper_plugin, 'groupId').text = MAVEN_BUILD_HELPER_GROUP_ID
        et.SubElement(build_helper_plugin, 'artifactId').text = MAVEN_BUILD_HELPER_ARTIFACT_ID
        et.SubElement(build_helper_plugin, 'version').text = MAVEN_BUILD_HELPER_VERSION
        executions = et.SubElement(build_helper_plugin, 'executions')
        if source_dirs:
            execution = et.SubElement(executions, 'execution')
            et.SubElement(execution, 'id').text = 'add-source'
            et.SubElement(execution, 'phase').text = 'generate-sources'
            et.SubElement(et.SubElement(execution, 'goals'), 'goal').text = 'add-source'
            sources = et.SubElement(et.SubElement(execution, 'configuration'), 'sources')
            for source_dir in source_dirs:
                et.SubElement(sources, 'source').text = source_dir
        if external_jars:
            execution = et.SubElement(executions, 'execution')
            et.SubElement(execution, 'id').text = 'attach-artifacts'
            et.SubElement(execution, 'phase').text = 'generate-sources'
            et.SubElement(et.SubElement(execution, 'goals'), 'goal').text = 'attach-artifact'
            artifacts = et.SubElement(et.SubElement(execution, 'configuration'), 'artifacts')
            for external_jar in external_jars:
                external_artifact = et.SubElement(artifacts, 'artifact')
                et.SubElement(external_artifact, 'file').text = '${basedir}/' + external_jar
                et.SubElement(external_artifact, 'type').text = 'jar'
        if test_source_dirs:
            execution = et.SubElement(executions, 'execution')
            et.SubElement(execution, 'id').text = 'add-test-source'
            et.SubElement(execution, 'phase').text = 'generate-test-sources'
            et.SubElement(et.SubElement(execution, 'goals'), 'goal').text = 'add-test-source'
            sources = et.SubElement(et.SubElement(execution, 'configuration'), 'sources')
            for source_dir in source_dirs:
                et.SubElement(sources, 'source').text = source_dir

    if run_java_programs:
        exec_plugin = et.SubElement(plugins, 'plugin')
        et.SubElement(exec_plugin, 'groupId').text = MAVEN_EXEC_GROUP_ID
        et.SubElement(exec_plugin, 'artifactId').text = MAVEN_EXEC_ARTIFACT_ID
        et.SubElement(exec_plugin, 'version').text = MAVEN_EXEC_VERSION
        jp_dependencies = et.SubElement(exec_plugin, 'dependencies')
        executions = et.SubElement(exec_plugin, 'executions')
        for java_program in run_java_programs:
            execution = et.SubElement(executions, 'execution')
            et.SubElement(execution, 'phase').text = 'generate-sources'
            et.SubElement(et.SubElement(execution, 'goals'), 'goal').text = 'java'
            jp_configuration = et.SubElement(execution, 'configuration')
            main_cls, args = None, []
            for word in java_program['cmd']:
                if not main_cls and not word.startswith('-'):
                    main_cls = word
                else:
                    args.append(word)
            et.SubElement(jp_configuration, 'mainClass').text = main_cls
            et.SubElement(jp_configuration, 'includePluginDependencies').text = 'true'
            et.SubElement(jp_configuration, 'includeProjectDependencies').text = 'false'
            if args:
                jp_arguments = et.SubElement(jp_configuration, 'arguments')
                for arg in args:
                    et.SubElement(jp_arguments, 'argument').text = arg
            if java_program['deps']:
                for jp_dep in java_program['deps']:
                    jp_dependency = et.SubElement(jp_dependencies, 'dependency')
                    jp_g, jp_a, jp_v = jp_dep.split(':')
                    et.SubElement(jp_dependency, 'groupId').text = jp_g
                    et.SubElement(jp_dependency, 'artifactId').text = jp_a
                    et.SubElement(jp_dependency, 'version').text = jp_v
                    et.SubElement(jp_dependency, 'type').text = 'jar'

    if target_dependencies + test_target_dependencies:
        dependencies = et.SubElement(project, 'dependencies')
        for target_dependency in target_dependencies + test_target_dependencies:
            dependency = et.SubElement(dependencies, 'dependency')
            dependency_info = split_artifacts(target_dependency)

            group_id, artifact_id, version, classifier = dependency_info[0].split(':')

            et.SubElement(dependency, 'groupId').text = group_id
            et.SubElement(dependency, 'artifactId').text = artifact_id
            et.SubElement(dependency, 'version').text = version
            if classifier:
                et.SubElement(dependency, 'classifier').text = classifier
            if target_dependency in test_target_dependencies:
                et.SubElement(dependency, 'scope').text = 'test'

            if len(dependency_info) > 1:
                exclusions = et.SubElement(dependency, 'exclusions')
                for exclude in dependency_info[1:]:
                    group_id, artifact_id = exclude.split(':')
                    exclusion_el = et.SubElement(exclusions, 'exclusion')
                    et.SubElement(exclusion_el, 'groupId').text = group_id
                    et.SubElement(exclusion_el, 'artifactId').text = artifact_id

    if test_target_dependencies_exclude:
        surefire_plugin = et.SubElement(plugins, 'plugin')
        et.SubElement(surefire_plugin, 'groupId').text = MAVEN_SUREFIRE_GROUP_ID
        et.SubElement(surefire_plugin, 'artifactId').text = MAVEN_SUREFIRE_ARTIFACT_ID
        et.SubElement(surefire_plugin, 'version').text = MAVEN_SUREFIRE_VERSION
        classpath_excludes = et.SubElement(et.SubElement(surefire_plugin, 'configuration'), 'classpathDependencyExcludes')
        for classpath_exclude in test_target_dependencies_exclude:
            et.SubElement(classpath_excludes, 'classpathDependencyExclude').text = classpath_exclude

    et.register_namespace('', DEFAULT_NAMESPACE)
    et.register_namespace('xsi', XSI_NAMESPACE)

    _indent(project)

    et.ElementTree(project).write(pom_path)
    sys.stderr.write("[MAVEN EXPORT] Generated {} file for target {}\n".format(os.path.basename(pom_path), target_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-path', action='store', default='')
    parser.add_argument('--target', action='store')
    parser.add_argument('--pom-path', action='store')
    parser.add_argument('--source-dirs', action='append', default=[])
    parser.add_argument('--external-jars', action='append', default=[])
    parser.add_argument('--resources', action='append', default=[])
    parser.add_argument('--run-java-programs', action='append', default=[])
    parser.add_argument('--output-dir')
    parser.add_argument('--final-name')
    parser.add_argument('--packaging', default='jar')
    parser.add_argument('--target-dependencies', action='append', default=[])
    parser.add_argument('--test-target-dependencies', action='append', default=[])
    parser.add_argument('--test-target-dependencies-exclude', action='append', default=[])
    parser.add_argument('--modules-path', action='store')
    parser.add_argument('--properties')
    parser.add_argument('--test-source-dirs', action='append', default=[])
    parser.add_argument('--test-resource-dirs', action='append', default=[])
    args = parser.parse_args()

    build_pom_and_export_to_maven(**vars(args))
