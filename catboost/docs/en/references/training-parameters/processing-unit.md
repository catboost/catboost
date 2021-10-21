# Processing unit settings

## task_type {#task_type}

Command line: `--task-type`

#### Description

The processing unit type to use for training.

Possible values:
- CPU
- GPU

**Type**

{{ python-type--string }}

**Default value**

{{ fit__python-r__calcer_type }}

**Supported processing units**

{{ cpu-gpu }}


## devices {#devices}

Command line: `--devices`

#### Description

IDs of the GPU devices to use for training (indices are zero-based).

Format

- `<unit ID>` for one device (for example, `3`)
- `<unit ID1>:<unit ID2>:..:<unit IDN>` for multiple devices (for example, `devices='0:1:3'`)
- `<unit ID1>-<unit IDN>` for a range of devices (for example, `devices='0-3'`)

**Type**

{{ python-type--string }}

**Default value**

{% cut "Python package" %}

{{ fit__python-r__device_config }}

{% endcut %}

{% cut "R package" %}

-1 (all GPU devices are used if the corresponding processing unit type is selected)

{% endcut %}

{% cut "Command-line" %}

-1 (use all devices)

{% endcut %}

**Supported processing units**

{{ calcer_type__cpu }}
