import os

version = os.environ.get('CATBOOST_PACKAGE_VERSION')
counter = os.environ.get('GO_PIPELINE_COUNTER', '0')
if version:
    with open('catboost/python-package/catboost/version.py', 'w') as version_file:
        version_file.write('VERSION = \'{}.dev{}\'\n'.format(version, counter))
