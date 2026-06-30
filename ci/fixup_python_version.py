import os

version = os.environ.get('CATBOOST_PACKAGE_VERSION')
if version:
    github_short_sha = os.environ.get('GITHUB_SHORT_SHA')
    if github_short_sha:
        full_version = version + '.dev1+g' + github_short_sha
    else:
        counter = os.environ.get('GO_PIPELINE_COUNTER', '0')
        full_version = version + '.dev' + counter

    with open('catboost/python-package/catboost/version.py', 'w') as version_file:
        version_file.write('VERSION = \'{}\'\n'.format(full_version))
