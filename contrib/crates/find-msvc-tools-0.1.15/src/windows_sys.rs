use crate::windows_sys_bindings as win;

// Private API
pub(crate) use win::*;

// Suppress unused warnings for types that can't be excluded from the bindings
#[allow(unused)]
use win::{IID_IUnknown as _, IUnknown_Vtbl as _};

// Public only for cc-rs
pub const FILE_ATTRIBUTE_TEMPORARY: u32 = win::FILE_ATTRIBUTE_TEMPORARY as u32;
pub const INFINITE: u32 = win::INFINITE;
pub const MAXIMUM_WAIT_OBJECTS: usize = win::MAXIMUM_WAIT_OBJECTS as usize;
pub const SYNCHRONIZE: u32 = win::SYNCHRONIZE as u32;
pub const WAIT_FAILED: u32 = win::WAIT_FAILED;
pub const WAIT_OBJECT_0: u32 = win::WAIT_OBJECT_0 as u32;
pub use win::{OpenSemaphoreA, PeekNamedPipe, WaitForMultipleObjects};
