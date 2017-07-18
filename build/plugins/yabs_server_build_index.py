# -*- coding: utf-8 -*-

import os

from _common import sort_by_keywords, stripext

UC = 'tools/uc'
MAKE_SQLITE_DB = 'yabs/server/test/tools/make_sqlite_db'
MAKE_YSON_FROM_SQLITE = 'yabs/server/test/tools/make_yson_from_sqlite'
MAKE_REQUESTS = 'yabs/server/test/tools/requests_from_qxl'
MKDBLM = 'yabs/server/tools/mkdblm'
MKDBDSSM = 'yabs/server/tools/mkdbdssm'
YT_MKDB = 'yabs/server/cs/tools/yt_mkdb'
BUILD_YSON_TEST_BASE = 'yabs/server/cs/tools/yt_mkdb/build_test_base.py'
BUILD_LM_TEST_BASE = 'yabs/server/tools/mkdblm/build_test_base.py'
BUILD_DSSM_TEST_BASE = 'yabs/server/tools/mkdbdssm/build_test_base.py'
MKDB_INFO_JSON = 'yabs/server/tools/mkdb_info/generated/mkdb_info.json'
CREDEFS_JSON = 'yabs/server/test/data/credefs.json'

DEFAULT_STAGE = 1
DEFAULT_TIMESTAMP = 1449779974
CODEC = 'zstd_5'


def get_stgroup(stat):
    stgroup = int(stat) % 18
    if stgroup == 0:
        stgroup = 18
    return stgroup


def get_tags(stats):
    if stats is None:
        stats = [i for i in xrange(1, 150) if i not in xrange(91, 100)]

    stgroups = sorted(set(get_stgroup(i) for i in stats))

    tags = (
        ['dbw', 'dbc', 'dbm1', 'dbe', 'dbis', 'dbds', 'order_update',
         'geo_segments', 'lua_templates', 'ssp_moderation', 'region', 'network', 'constant', 'reverse_geocoder',
         'domain', 'formula', 'search_domain_factors', 'query_factors', 'dsp_creative',
         'weather', 'adaptive', 'auction_phrase', 'search_query', 'qtail_linear_model'] +
        ['banner_update_%02d' % int(i) for i in stgroups] +
        ['st%03d' % int(i) for i in stats] +
        ['phrase_price_%03d' % int(i) for i in stats]
    )
    lm_tags = ['dblm_0'] + ['dblm_%02d' % i for i in stgroups]
    dssm_tags = ['dssm_%02d' % i for i in stgroups]

    return tags, lm_tags, dssm_tags


def onyabs_server_prepare_qxl_from_sandbox(unit, *args):
    (resource_id, qxl_name), kv = sort_by_keywords(
        {'TIMESTAMP': 1},
        args
    )

    json_basename = stripext(qxl_name)
    prefix = unit.path().replace('$S', '${ARCADIA_BUILD_ROOT}') + '/'
    pickle_name = prefix + json_basename + '.pickle'
    enabled_stats_name = prefix + json_basename + '.sts'

    timestamp = int(kv.get('TIMESTAMP', [DEFAULT_TIMESTAMP])[0])

    unit.onfrom_sandbox(['FILE', resource_id, 'OUT', prefix + qxl_name])

    unit.onrun_program([
        MAKE_SQLITE_DB,
        '--credefs', CREDEFS_JSON,
        '--qxl', prefix + qxl_name,
        '--pickle', pickle_name,
        '--nowtime', str(timestamp - 1),
        '--stenabled', enabled_stats_name,
        'IN', CREDEFS_JSON,
        'IN', prefix + qxl_name,
        'OUT', pickle_name,
        'OUT', enabled_stats_name
    ])

    request_dumps = [[
        'yabs_test.json',
    ], [
        'simulator.json',
    ], [
        'http_laas.json',
        'http_metrika.json',
        'http_metasearch.json',
        'http_metapartner.json',
        'http_metarank.json',
    ]]

    for group in request_dumps:
        args = [
            MAKE_REQUESTS,
            prefix + qxl_name,
        ]
        args += group
        args += ['IN', prefix + qxl_name]
        for fname in group:
            args += ['OUT', fname]
        unit.onrun_program(args)


def onyabs_server_build_yson_index(unit, *args):
    (qxl_dir, json_basename), kv = sort_by_keywords(
        {'STATS': -1, 'TIMESTAMP': 1, 'STAGE': -1},
        args
    )

    tags, lm_tags, dssm_tags = get_tags(kv.get('STATS'))

    qxl_prefix = '${ARCADIA_BUILD_ROOT}/' + qxl_dir + '/'
    pickle_name = qxl_prefix + json_basename + '.pickle'

    prefix = unit.path().replace('$S', '${ARCADIA_BUILD_ROOT}') + '/'

    stage = int(kv.get('STAGE', [DEFAULT_STAGE])[0])
    timestamp = int(kv.get('TIMESTAMP', [DEFAULT_TIMESTAMP])[0])

    deps = [
        qxl_dir,
        os.path.dirname(MKDB_INFO_JSON),
    ]

    unit.onpeerdir(deps)
    unit.ondepends(deps)

    for tag in tags:
        yson_name = prefix + tag + '.yson'
        meta_yson_name = prefix + tag + '.meta.yson'
        db_name = prefix + tag + '.yabs.' + CODEC

        unit.onrun_program([
            MAKE_YSON_FROM_SQLITE,
            '--db', pickle_name,
            '--yson', yson_name,
            '--meta_yson', meta_yson_name,
            '--tag', tag,
            '--stage', str(stage),
            '--nowtime', str(timestamp),
            '--mkdb_info', '${ARCADIA_BUILD_ROOT}/' + MKDB_INFO_JSON,
            'IN', pickle_name,
            'IN', '${ARCADIA_BUILD_ROOT}/' + MKDB_INFO_JSON,
            'OUT', yson_name,
            'OUT', meta_yson_name,
        ])

        # FIXME: path substitution for TOOLs is bugged,
        # you cannot pass two different TOOLs as arguments
        unit.onbuiltin_python([
            BUILD_YSON_TEST_BASE,
            '--mkdb', YT_MKDB,
            '--yson', yson_name,
            '--output', db_name,
            '--tag', tag,
            '--timestamp', str(timestamp),
            '--compress', CODEC,
            'IN', yson_name,
            'OUT', db_name,
            'TOOL', YT_MKDB,
            'TOOL', UC,
        ])

    for tag in lm_tags:
        db_name = prefix + tag + '.yabs.' + CODEC

        unit.onbuiltin_python([
            BUILD_LM_TEST_BASE,
            '--mkdblm', MKDBLM,
            '--output', db_name,
            '--tag', tag,
            '--timestamp', str(timestamp),
            '--compress', CODEC,
            'OUT', db_name,
            'TOOL', MKDBLM,
            'TOOL', UC,
        ])

    for tag in dssm_tags:
        db_name = prefix + tag + '.yabs.' + CODEC

        unit.onbuiltin_python([
            BUILD_DSSM_TEST_BASE,
            '--mkdbdssm', MKDBDSSM,
            '--output', db_name,
            '--tag', tag,
            '--compress', CODEC,
            'OUT', db_name,
            'TOOL', MKDBDSSM,
            'TOOL', UC,
        ])
