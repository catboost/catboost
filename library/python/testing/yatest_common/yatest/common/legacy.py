from . import canonical


def old_canonical_file(output_file_name, storage_md5):
    import yalibrary.svn
    yalibrary.svn.run_svn([
        'export',
        'svn+ssh://arcadia.yandex.ru/arc/trunk/arcadia_tests_data/tests_canonical_output/' + storage_md5,
        output_file_name,
        "--force"
    ])
    return canonical.canonical_file(output_file_name)
