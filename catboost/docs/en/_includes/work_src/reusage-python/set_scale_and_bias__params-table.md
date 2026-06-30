### scale

#### Description

The model scale.

{% include [get_scale_and_bias-scale_and_bias__desc](scale_and_bias__desc.md) %}

The value of this parameters affects the prediction by changing the default value of the scale.

**Possible types**

{{ python-type--float }}

**Default value**

1

### bias

#### Description

The model bias.

{% include [get_scale_and_bias-scale_and_bias__desc](scale_and_bias__desc.md) %}

The value of this parameters affects the prediction by changing the default value of the bias.

**Possible types**

{{ python-type--float }}

**Default value**

Depends on the value of the `--boost-from-average` for the Command-line version parameter:

- True — The best constant value for the specified loss function
- False — 0
