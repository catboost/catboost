use alloc::{Allocator, SliceWrapper, SliceWrapperMut};

use brotli_decompressor::ffi::alloc_util::SubclassableAllocator;

use crate::enc::BrotliAlloc;

pub struct BrotliSubclassableAllocator(SubclassableAllocator);

impl BrotliSubclassableAllocator {
    pub fn new(s: SubclassableAllocator) -> BrotliSubclassableAllocator {
        BrotliSubclassableAllocator(s)
    }
}

#[derive(Default)]
pub struct SendableMemoryBlock<T: Clone + Default>(
    <SubclassableAllocator as Allocator<T>>::AllocatedMemory,
);
impl<T: Clone + Default> SliceWrapperMut<T> for SendableMemoryBlock<T> {
    fn slice_mut(&mut self) -> &mut [T] {
        self.0.slice_mut()
    }
}
impl<T: Clone + Default> SliceWrapper<T> for SendableMemoryBlock<T> {
    fn slice(&self) -> &[T] {
        self.0.slice()
    }
}
impl<T: Clone + Default> Allocator<T> for BrotliSubclassableAllocator {
    type AllocatedMemory = SendableMemoryBlock<T>;
    fn alloc_cell(&mut self, s: usize) -> Self::AllocatedMemory {
        SendableMemoryBlock(self.0.alloc_cell(s))
    }
    fn free_cell(&mut self, data: Self::AllocatedMemory) {
        self.0.free_cell(data.0)
    }
}

impl BrotliAlloc for BrotliSubclassableAllocator {}
#[cfg(not(feature = "safe"))]
unsafe impl Send for BrotliSubclassableAllocator {}

#[cfg(not(feature = "safe"))]
unsafe impl<T: Clone + Default> Send for SendableMemoryBlock<T> {}

#[cfg(not(feature = "std"))]
#[cfg(feature = "no-stdlib-ffi-binding")]
#[panic_handler]
extern "C" fn panic_impl(_: &::core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(not(feature = "std"))]
#[cfg(feature = "no-stdlib-ffi-binding")]
#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
