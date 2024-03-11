# Comm

It provides a way to register a Kernel Comm implementation, as per the Jupyter kernel protocol.
It also provides a base Comm implementation and a default CommManager that can be used.

## Register a comm implementation in the kernel:

### Case 1: Using the default CommManager and the BaseComm implementations

We provide default implementations for usage in IPython:

```python
import comm


class MyCustomComm(comm.base_comm.BaseComm):
    def publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **keys):
        # TODO implement the logic for sending comm messages through the iopub channel
        pass


comm.create_comm = MyCustomComm
```

This is typically what ipykernel and JupyterLite's pyolite kernel will do.

### Case 2: Providing your own comm manager creation implementation

```python
import comm

comm.create_comm = custom_create_comm
comm.get_comm_manager = custom_comm_manager_getter
```

This is typically what xeus-python does (it has its own manager implementation using xeus's C++ messaging logic).

## Comm users

Libraries like ipywidgets can then use the comms implementation that has been registered by the kernel:

```python
from comm import create_comm, get_comm_manager

# Create a comm
comm_manager = get_comm_manager()
comm = create_comm()

comm_manager.register_comm(comm)
```
