use core;
#[cfg(feature="std")]
use std;
use ::alloc;
use super::interface::{c_void, CAllocator};
#[cfg(feature="std")]
use std::vec::Vec;
#[cfg(feature="std")]
pub use std::boxed::Box;

#[cfg(feature="std")]
pub struct MemoryBlock<Ty:Sized+Default>(Box<[Ty]>);
#[cfg(feature="std")]
impl<Ty:Sized+Default> Default for MemoryBlock<Ty> {
    fn default() -> Self {
        MemoryBlock(Vec::<Ty>::new().into_boxed_slice())
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default> alloc::SliceWrapper<Ty> for MemoryBlock<Ty> {
    fn slice(&self) -> &[Ty] {
        &self.0[..]
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default> alloc::SliceWrapperMut<Ty> for MemoryBlock<Ty> {
    fn slice_mut(&mut self) -> &mut [Ty] {
        &mut self.0[..]
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default> core::ops::Index<usize> for MemoryBlock<Ty> {
    type Output = Ty;
    fn index(&self, index:usize) -> &Ty {
        &self.0[index]
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default> core::ops::IndexMut<usize> for MemoryBlock<Ty> {

    fn index_mut(&mut self, index:usize) -> &mut Ty {
        &mut self.0[index]
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default> Drop for MemoryBlock<Ty> {
    fn drop (&mut self) {
        if self.0.len() != 0 {
            print!("leaking memory block of length {} element size: {}\n", self.0.len(), core::mem::size_of::<Ty>());

            let to_forget = core::mem::replace(self, MemoryBlock::default());
            core::mem::forget(to_forget);// leak it -- it's the only safe way with custom allocators
        }
    }
}
pub struct SubclassableAllocator {
    alloc: CAllocator
    // have alternative ty here
}

impl SubclassableAllocator {
    pub unsafe fn new(sub_alloc:CAllocator) -> Self {
        SubclassableAllocator{
            alloc:sub_alloc,
        }
    }
}
#[cfg(feature="std")]
impl<Ty:Sized+Default+Clone> alloc::Allocator<Ty> for SubclassableAllocator {
    type AllocatedMemory = MemoryBlock<Ty>;
    fn alloc_cell(&mut self, size:usize) ->MemoryBlock<Ty>{
        if size == 0 {
            return MemoryBlock::<Ty>::default();
        }
        if let Some(alloc_fn) = self.alloc.alloc_func {
            let ptr = alloc_fn(self.alloc.opaque, size * core::mem::size_of::<Ty>());
            let typed_ptr = unsafe {core::mem::transmute::<*mut c_void, *mut Ty>(ptr)};
            let slice_ref = unsafe {super::slice_from_raw_parts_or_nil_mut(typed_ptr, size)};
            for item in slice_ref.iter_mut() {
                unsafe{core::ptr::write(item, Ty::default())};
            }
            return MemoryBlock(unsafe{Box::from_raw(slice_ref)})
        }
        MemoryBlock(vec![Ty::default();size].into_boxed_slice())
    }
    fn free_cell(&mut self, mut bv:MemoryBlock<Ty>) {
        if (*bv.0).len() != 0 {
            if let Some(_) = self.alloc.alloc_func {
                let slice_ptr = (*bv.0).as_mut_ptr();
                let _box_ptr = Box::into_raw(core::mem::replace(&mut bv.0, Vec::<Ty>::new().into_boxed_slice()));
                if let Some(free_fn) = self.alloc.free_func {
                    unsafe {free_fn(self.alloc.opaque, core::mem::transmute::<*mut Ty, *mut c_void>(slice_ptr))};
                }
            } else {
                let _to_free = core::mem::replace(&mut bv.0, Vec::<Ty>::new().into_boxed_slice());
            }
        }
    }
}











#[cfg(not(feature="std"))]
static mut G_SLICE:&'static mut[u8] = &mut[];
#[cfg(not(feature="std"))]
pub struct MemoryBlock<Ty:Sized+Default>(*mut[Ty]);
#[cfg(not(feature="std"))]
impl<Ty:Sized+Default> Default for MemoryBlock<Ty> {
    fn default() -> Self {
        MemoryBlock(unsafe{core::mem::transmute::<*mut [u8], *mut[Ty]>(G_SLICE.as_mut())})
    }
}
#[cfg(not(feature="std"))]
impl<Ty:Sized+Default> alloc::SliceWrapper<Ty> for MemoryBlock<Ty> {
    fn slice(&self) -> &[Ty] {
        if unsafe{(*self.0).len()} == 0 {
            &[]
        } else {
            unsafe{super::slice_from_raw_parts_or_nil(&(*self.0)[0], (*self.0).len())}
        }
    }
}
#[cfg(not(feature="std"))]
impl<Ty:Sized+Default> alloc::SliceWrapperMut<Ty> for MemoryBlock<Ty> {
    fn slice_mut(&mut self) -> &mut [Ty] {
        if unsafe{(*self.0).len()} == 0 {
            &mut []
        } else {
            unsafe{super::slice_from_raw_parts_or_nil_mut(&mut (*self.0)[0], (*self.0).len())}
        }
    }
}

#[cfg(not(feature="std"))]
#[cfg(feature="no-stdlib-ffi-binding")]
#[panic_handler]
extern fn panic_impl(_: &::core::panic::PanicInfo) -> ! {
    loop {}
}
#[cfg(not(feature="std"))]
#[cfg(feature="no-stdlib-ffi-binding")]
#[lang = "eh_personality"]
extern "C" fn eh_personality() {
}

#[cfg(not(feature="std"))]
impl<Ty:Sized+Default> core::ops::Index<usize> for MemoryBlock<Ty> {
    type Output = Ty;
    fn index(&self, index:usize) -> &Ty {
        unsafe{&(*self.0)[index]}
    }
}
#[cfg(not(feature="std"))]
impl<Ty:Sized+Default> core::ops::IndexMut<usize> for MemoryBlock<Ty> {

    fn index_mut(&mut self, index:usize) -> &mut Ty {
        unsafe{&mut (*self.0)[index]}
    }
}

#[cfg(not(feature="std"))]
impl<Ty:Sized+Default+Clone> alloc::Allocator<Ty> for SubclassableAllocator {
    type AllocatedMemory = MemoryBlock<Ty>;
    fn alloc_cell(&mut self, size:usize) ->MemoryBlock<Ty>{
        if size == 0 {
            return MemoryBlock::<Ty>::default();
        }
        if let Some(alloc_fn) = self.alloc.alloc_func {
            let ptr = alloc_fn(self.alloc.opaque, size * core::mem::size_of::<Ty>());
            let typed_ptr = unsafe {core::mem::transmute::<*mut c_void, *mut Ty>(ptr)};
            let slice_ref = unsafe {super::slice_from_raw_parts_or_nil_mut(typed_ptr, size)};
            for item in slice_ref.iter_mut() {
                unsafe{core::ptr::write(item, Ty::default())};
            }
            return MemoryBlock(slice_ref.as_mut())
        } else {
            panic!("Must provide allocators in no-stdlib code");
        }
    }
    fn free_cell(&mut self, mut bv:MemoryBlock<Ty>) {
        use alloc::SliceWrapper;
        use alloc::SliceWrapperMut;
        if bv.slice().len() != 0 {
            if let Some(_) = self.alloc.alloc_func {
                if let Some(free_fn) = self.alloc.free_func {
                    unsafe {free_fn(self.alloc.opaque, core::mem::transmute::<*mut Ty, *mut c_void>(&mut bv.slice_mut()[0]))};
                }
                let _ = core::mem::replace(&mut bv,
                                           MemoryBlock::<Ty>::default());
            } else {
                panic!("Must provide allocators in no-stdlib code");
            }
        }
    }
}


#[cfg(not(feature="std"))]
pub fn free_stdlib<T>(_data: *mut T, _size: usize) {
    panic!("Must supply allocators if calling divans when compiled with features=no-stdlib");
}
#[cfg(not(feature="std"))]
pub fn alloc_stdlib<T:Sized+Default+Copy+Clone>(_size: usize) -> *mut T {
    panic!("Must supply allocators if calling divans when compiled with features=no-stdlib");
}

#[cfg(feature="std")]
pub unsafe fn free_stdlib<T>(ptr: *mut T, size: usize) {
    let slice_ref = super::slice_from_raw_parts_or_nil_mut(ptr, size);
    let _ = Box::from_raw(slice_ref); // free on drop
}
#[cfg(feature="std")]
pub fn alloc_stdlib<T:Sized+Default+Copy+Clone>(size: usize) -> *mut T {
    std::panic::catch_unwind(|| {
        let mut newly_allocated = vec![T::default();size].into_boxed_slice();
        let slice_ptr = newly_allocated.as_mut_ptr();
        let _box_ptr = Box::into_raw(newly_allocated);
        slice_ptr
    }).unwrap_or(core::ptr::null_mut())
}
