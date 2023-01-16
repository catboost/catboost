
Compile the packages using one of the following methods:
- Use one of the Python versions provided by the `ya make` utility:
    
    ```bash
    ../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DUSE_SYSTEM_PYTHON=<Python version> [optional parameters]
    ```
    
- Use one of the Python versions installed on the machine:
    
    ```python
    ../../../ya make -r -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=<path to the required python-config> [optional parameters]
    ```
    
<table>
  <tr>
    <th>Parameter</th>
    <th>Description</th>
  </tr>
  <tr>
    <td colspan="2"><b>Parameters that define the Python version to use for compiling. Only one of the following blocks of options can be used at a time</b></td>
  </tr>
  <tr>
    <td colspan="2">Use one of the Python versions provided by the <code>ya make</code> utility</td>
  </tr>
  <tr>
     <td><code>-DUSE_SYSTEM_PYTHON</code></td><td>The version of Python to use for compiling the package on machines without installed Python.

The following Python versions are supported and can be defined as values for this parameter:
- 2.7
- 3.4
- 3.5
- 3.6</td>
  </tr>
  <tr>
    <td colspan="2">Use one of the Python versions installed on the machine</td>
  </tr>
  <tr>
     <td><code>-DPYTHON_CONFIG</code></td><td>Defines the path to the configuration of an installed Python version to use for compiling the package.

Value examples:
- <code>python2-config</code> for Python 2
- <code>python3-config</code> for Python 3
- <code>/usr/bin/python2.7-config</code>

<note type="attention">

- The configuration must be explicitly named <code>python3-config</code> to successfully build the package for Python 3.
- Manually redefine the following variables when encountering problems with the Python configuration:
    - <code>-DPYTHON_INCLUDE</code>
    - <code>-DPYTHON_LIBRARIES</code>
    - <code>-DPYTHON_LDFLAGS</code>
    - <code>-DPYTHON_FLAGS</code>
    - <code>-DPYTHON_BIN</code>

</note></td>
  </tr>
  <tr>
    <td colspan="2"><b>Optional parameters</b></td>
  </tr>
  <tr>
     <td><code>-DCUDA_ROOT</code></td><td>The path to CUDA. This parameter is required to support training on GPU.</td>
  </tr>
  <tr>
     <td><code>-DHAVE_CUDA=no</code></td><td>Disable CUDA support. This speeds up compilation.

By default, the package is built with CUDA support if CUDA Toolkit is installed.</td>
  </tr>
  <tr>
     <td><code>-o</code></td><td>The directory to output the compiled package to. By default, the current directory is used.</td>
  </tr>
</table>
