import _common as common
import ymake
import os


def onios_assets(unit, *args):
    _, kv = common.sort_by_keywords(
        {'ROOT': 1, 'CONTENTS': -1, 'FLAGS': -1},
        args
    )
    if not kv.get('ROOT', []) and kv.get('CONTENTS', []):
        ymake.report_configure_error('Please specify ROOT directory for assets')
    origin_root = kv.get('ROOT')[0]
    destination_root = os.path.normpath(os.path.join('$BINDIR', os.path.basename(origin_root)))
    rel_list = []
    for cont in kv.get('CONTENTS', []):
        rel = os.path.relpath(cont, origin_root)
        if rel.startswith('..'):
            ymake.report_configure_error('{} is not subpath of {}'.format(cont, origin_root))
        rel_list.append(rel)
    if not rel_list:
        return
    results_list = [os.path.join('$B', unit.path()[3:], os.path.basename(origin_root), i) for i in rel_list]
    if len(kv.get('CONTENTS', [])) != len(results_list):
        ymake.report_configure_error('IOS_ASSETTS content length is not equals results')
    for s, d in zip(kv.get('CONTENTS', []), results_list):
        unit.oncopy_file([s, d])
    if kv.get('FLAGS', []):
        unit.onios_app_assets_flags(kv.get('FLAGS', []))
    unit.on_ios_assets([destination_root] + results_list)
