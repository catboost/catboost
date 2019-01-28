import argparse
import json
import os
import subprocess
import sys


def just_do_it():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["create", "spawn", "kill"])
    parser.add_argument("--simctl", help="simctl binary path")
    parser.add_argument("--profiles", help="profiles path")
    parser.add_argument("--device-dir", help="devices directory")
    parser.add_argument("--device-name", help="temp device name")
    args, tail = parser.parse_known_args()
    if args.action == 'create':
        action_create(args.simctl, args.profiles, args.device_dir, args.device_name, tail)
    elif args.action == "spawn":
        action_spawn(args.simctl, args.profiles, args.device_dir, args.device_name, tail)
    elif args.action == "kill":
        action_kill(args.simctl, args.profiles, args.device_dir, args.device_name)


def action_create(simctl, profiles, device_dir, name, args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-type", default="com.apple.CoreSimulator.SimDeviceType.iPhone-X")
    parser.add_argument("--device-runtime", default="com.apple.CoreSimulator.SimRuntime.iOS-12-1")
    args = parser.parse_args(args)
    all_devices = list(get_all_devices(simctl, profiles, device_dir))
    if filter(lambda x: x["name"] == name, all_devices):
        raise Exception("Device named {} already exists".format(name))
    subprocess.check_call([simctl, "--profiles", profiles, "--set", device_dir, "create", name, args.device_type, args.device_runtime])
    created = filter(lambda x: x["name"] == name, get_all_devices(simctl, profiles, device_dir))
    if not created:
        raise Exception("Creation error: temp device named {} not found".format(name))
    created = created[0]
    if created["availability"] != "(available)":
        raise Exception("Creation error: temp device {} status is {} ((available) expected)".format(name, created["availability"]))


def action_spawn(simctl, profiles, device_dir, name, args):
    device = filter(lambda x: x["name"] == name, get_all_devices(simctl, profiles, device_dir))
    if not device:
        raise Exception("Can't spawn process: device named {} not found".format(name))
    if len(device) > 1:
        raise Exception("Can't spawn process: too many devices named {} found".format(name))
    device = device[0]
    os.execv(simctl, [simctl, "--profiles", profiles, "--set", device_dir, "spawn", device["udid"]] + args)


def action_kill(simctl, profiles, device_dir, name):
    device = filter(lambda x: x["name"] == name, get_all_devices(simctl, profiles, device_dir))
    if not device:
        print >> sys.stderr, "Device named {} not found; do nothing".format(name)
        return
    if len(device) > 1:
        raise Exception("Can't remove: too many devices named {}:\n{}".format(name, '\n'.join(i for i in device)))
    device = device[0]
    os.execv(simctl, [simctl, "--profiles", profiles, "--set", device_dir, "delete", device["udid"]])


def get_all_devices(simctl, profiles, device_dir):
    p = subprocess.Popen([simctl, "--profiles", profiles, "--set", device_dir, "list", "--json", "devices"], stdout=subprocess.PIPE)
    out, _ = p.communicate()
    rc = p.wait()
    if rc:
        raise Exception("Devices list command return code is {}\nstdout:\n{}".format(rc, out))
    raw_object = json.loads(out)
    if "devices" not in raw_object:
        raise Exception("Devices not found in\n{}".format(json.dumps(raw_object)))
    raw_object = raw_object["devices"]
    for os_name, devices in raw_object.items():
        for device in devices:
            device["os_name"] = os_name
            yield device


if __name__ == '__main__':
    just_do_it()
