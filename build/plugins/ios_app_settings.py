import _common as common
import ymake
import os

def onios_app_settings(unit, *args):
    tail, kv = common.sort_by_keywords(
        {'OS_VERSION': 1, 'DEVICES': -1},
        args
    )
    if tail:
        ymake.report_configure_error('Bad IOS_COMMON_SETTINGS usage - unknown data: ' + str(tail))
    if kv.get('OS_VERSION', []):
        unit.onios_app_common_flags(['--minimum-deployment-target', kv.get('OS_VERSION', [])[0]])
        unit.onios_app_assets_flags(['--filter-for-device-os-version', kv.get('OS_VERSION', [])[0]])
    devices_flags = []
    for device in kv.get('DEVICES', []):
        devices_flags += ['--target-device', device]
    if devices_flags:
        unit.onios_app_common_flags(devices_flags)
