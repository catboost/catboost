from os.path import basename, splitext


def onbuild_mns_files(unit, *args):
    files = []
    name = ''
    ranking_suffix = ''
    check = ''
    index = 0
    fml_unused_tool = ''
    while index < len(args):
        if args[index] == 'NAME':
            index += 1
            name = args[index]
        elif args[index] == 'RANKING_SUFFIX':
            index += 1
            ranking_suffix = args[index]
        elif args[index] == 'CHECK':
            check = 'CHECK'
            fml_unused_tool = unit.get('FML_UNUSED_TOOL') or '$FML_UNUSED_TOOL'
        else:
            files.append(args[index])
        index += 1

    for filename in files:
        file_basename, _ = splitext(basename(filename))
        asmdataname = "staticMn{0}{1}Ptr".format(ranking_suffix, file_basename)
        output_name = 'mn.staticMn{0}{1}Ptr.cpp'.format(ranking_suffix, file_basename)
        unit.onbuild_mns_file([filename, name, output_name, ranking_suffix, check, fml_unused_tool, asmdataname])
