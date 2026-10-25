use super::super::alloc;
use super::super::alloc::{Allocator, SliceWrapper};
use crate::enc::combined_alloc::alloc_default;

pub struct BlockSplit<Alloc: alloc::Allocator<u8> + alloc::Allocator<u32>> {
    pub num_types: usize,
    pub num_blocks: usize,
    pub types: <Alloc as Allocator<u8>>::AllocatedMemory,
    pub lengths: <Alloc as Allocator<u32>>::AllocatedMemory,
}

impl<Alloc: alloc::Allocator<u8> + alloc::Allocator<u32>> Default for BlockSplit<Alloc> {
    fn default() -> Self {
        Self {
            num_types: 0,
            num_blocks: 0,
            types: alloc_default::<u8, Alloc>(),
            lengths: alloc_default::<u32, Alloc>(),
        }
    }
}

impl<Alloc: alloc::Allocator<u8> + alloc::Allocator<u32>> BlockSplit<Alloc> {
    pub fn new() -> BlockSplit<Alloc> {
        Self::default()
    }
    pub fn destroy(&mut self, m: &mut Alloc) {
        <Alloc as Allocator<u8>>::free_cell(m, core::mem::take(&mut self.types));
        <Alloc as Allocator<u32>>::free_cell(m, core::mem::take(&mut self.lengths));
        self.num_blocks = 0;
        self.num_types = 0;
    }
    pub fn types_alloc_size(&self) -> usize {
        self.types.slice().len()
    }
    pub fn lengths_alloc_size(&self) -> usize {
        self.lengths.slice().len()
    }
}
