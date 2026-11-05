pub use alloc::Allocator;

#[cfg(feature = "std")]
use alloc_stdlib::StandardAlloc;

use super::cluster::HistogramPair;
use super::command::Command;
use super::entropy_encode::HuffmanTree;
use super::hash_to_binary_tree::ZopfliNode;
use super::histogram::{ContextType, HistogramCommand, HistogramDistance, HistogramLiteral};
use super::interface::StaticCommand;
use super::util::floatX;
use super::{s16, v8, PDF};
/*
struct CombiningAllocator<T1, T2, AllocT1:Allocator<T1>, AllocT2:Allocator<T2>>(AllocT1, AllocT2);

impl <T1, T2, AllocT1:Allocator<T1>, AllocT2:Allocator<T2>> CombiningAllocator<T1, T2, AllocT1, AllocT2> {
  pub fn new(a: AllocT1, b: AllocT2) -> Self {
    CombiningAllocator(a, b)
  }
}

impl <T1, T2, AllocT1:Allocator<T1>, AllocT2:Allocator<T2>> Allocator<T1> for CombiningAllocator<T1, T2, AllocT1, AllocT2> {

}


impl <T1, T2, AllocT1:Allocator<T1>, AllocT2:Allocator<T2>> Allocator<T2> for CombiningAllocator<T1, T2, AllocT1, AllocT2> {

}
*/

pub trait BrotliAlloc:
    Allocator<u8>
    + Allocator<u16>
    + Allocator<i32>
    + Allocator<u32>
    + Allocator<u64>
    + Allocator<Command>
    + Allocator<super::util::floatX>
    + Allocator<v8>
    + Allocator<s16>
    + Allocator<PDF>
    + Allocator<StaticCommand>
    + Allocator<HistogramLiteral>
    + Allocator<HistogramCommand>
    + Allocator<HistogramDistance>
    + Allocator<HistogramPair>
    + Allocator<ContextType>
    + Allocator<HuffmanTree>
    + Allocator<ZopfliNode>
{
}

#[cfg(feature = "std")]
impl BrotliAlloc for StandardAlloc {}

pub struct CombiningAllocator<
    AllocU8: Allocator<u8>,
    AllocU16: Allocator<u16>,
    AllocI32: Allocator<i32>,
    AllocU32: Allocator<u32>,
    AllocU64: Allocator<u64>,
    AllocCommand: Allocator<Command>,
    AllocFloatX: Allocator<floatX>,
    AllocV8: Allocator<v8>,
    AllocS16: Allocator<s16>,
    AllocPDF: Allocator<PDF>,
    AllocStaticCommand: Allocator<StaticCommand>,
    AllocHistogramLiteral: Allocator<HistogramLiteral>,
    AllocHistogramCommand: Allocator<HistogramCommand>,
    AllocHistogramDistance: Allocator<HistogramDistance>,
    AllocHistogramPair: Allocator<HistogramPair>,
    AllocContextType: Allocator<ContextType>,
    AllocHuffmanTree: Allocator<HuffmanTree>,
    AllocZopfliNode: Allocator<ZopfliNode>,
> {
    alloc_u8: AllocU8,
    alloc_u16: AllocU16,
    alloc_i32: AllocI32,
    alloc_u32: AllocU32,
    alloc_u64: AllocU64,
    alloc_c: AllocCommand,
    alloc_f: AllocFloatX,
    alloc_f32x8: AllocV8,
    alloc_i16x16: AllocS16,
    alloc_pdf: AllocPDF,
    alloc_sc: AllocStaticCommand,
    alloc_hl: AllocHistogramLiteral,
    alloc_hc: AllocHistogramCommand,
    alloc_hd: AllocHistogramDistance,
    alloc_hp: AllocHistogramPair,
    alloc_ct: AllocContextType,
    alloc_ht: AllocHuffmanTree,
    alloc_zn: AllocZopfliNode,
}

impl<
        AllocU8: Allocator<u8>,
        AllocU16: Allocator<u16>,
        AllocI32: Allocator<i32>,
        AllocU32: Allocator<u32>,
        AllocU64: Allocator<u64>,
        AllocCommand: Allocator<Command>,
        AllocFloatX: Allocator<floatX>,
        AllocV8: Allocator<v8>,
        AllocS16: Allocator<s16>,
        AllocPDF: Allocator<PDF>,
        AllocStaticCommand: Allocator<StaticCommand>,
        AllocHistogramLiteral: Allocator<HistogramLiteral>,
        AllocHistogramCommand: Allocator<HistogramCommand>,
        AllocHistogramDistance: Allocator<HistogramDistance>,
        AllocHistogramPair: Allocator<HistogramPair>,
        AllocContextType: Allocator<ContextType>,
        AllocHuffmanTree: Allocator<HuffmanTree>,
        AllocZopfliNode: Allocator<ZopfliNode>,
    >
    CombiningAllocator<
        AllocU8,
        AllocU16,
        AllocI32,
        AllocU32,
        AllocU64,
        AllocCommand,
        AllocFloatX,
        AllocV8,
        AllocS16,
        AllocPDF,
        AllocStaticCommand,
        AllocHistogramLiteral,
        AllocHistogramCommand,
        AllocHistogramDistance,
        AllocHistogramPair,
        AllocContextType,
        AllocHuffmanTree,
        AllocZopfliNode,
    >
{
    pub fn new(
        alloc_u8: AllocU8,
        alloc_u16: AllocU16,
        alloc_i32: AllocI32,
        alloc_u32: AllocU32,
        alloc_u64: AllocU64,
        alloc_c: AllocCommand,
        alloc_f: AllocFloatX,
        alloc_f32x8: AllocV8,
        alloc_i16x16: AllocS16,
        alloc_pdf: AllocPDF,
        alloc_sc: AllocStaticCommand,
        alloc_hl: AllocHistogramLiteral,
        alloc_hc: AllocHistogramCommand,
        alloc_hd: AllocHistogramDistance,
        alloc_hp: AllocHistogramPair,
        alloc_ct: AllocContextType,
        alloc_ht: AllocHuffmanTree,
        alloc_zn: AllocZopfliNode,
    ) -> Self {
        CombiningAllocator {
            alloc_u8,
            alloc_u16,
            alloc_i32,
            alloc_u32,
            alloc_u64,
            alloc_c,
            alloc_f,
            alloc_f32x8,
            alloc_i16x16,
            alloc_pdf,
            alloc_sc,
            alloc_hl,
            alloc_hc,
            alloc_hd,
            alloc_hp,
            alloc_ct,
            alloc_ht,
            alloc_zn,
        }
    }
}

impl<
        AllocU8: Allocator<u8>,
        AllocU16: Allocator<u16>,
        AllocI32: Allocator<i32>,
        AllocU32: Allocator<u32>,
        AllocU64: Allocator<u64>,
        AllocCommand: Allocator<Command>,
        AllocFloatX: Allocator<floatX>,
        AllocV8: Allocator<v8>,
        AllocS16: Allocator<s16>,
        AllocPDF: Allocator<PDF>,
        AllocStaticCommand: Allocator<StaticCommand>,
        AllocHistogramLiteral: Allocator<HistogramLiteral>,
        AllocHistogramCommand: Allocator<HistogramCommand>,
        AllocHistogramDistance: Allocator<HistogramDistance>,
        AllocHistogramPair: Allocator<HistogramPair>,
        AllocContextType: Allocator<ContextType>,
        AllocHuffmanTree: Allocator<HuffmanTree>,
        AllocZopfliNode: Allocator<ZopfliNode>,
    > BrotliAlloc
    for CombiningAllocator<
        AllocU8,
        AllocU16,
        AllocI32,
        AllocU32,
        AllocU64,
        AllocCommand,
        AllocFloatX,
        AllocV8,
        AllocS16,
        AllocPDF,
        AllocStaticCommand,
        AllocHistogramLiteral,
        AllocHistogramCommand,
        AllocHistogramDistance,
        AllocHistogramPair,
        AllocContextType,
        AllocHuffmanTree,
        AllocZopfliNode,
    >
{
}

impl<
        AllocU8: Allocator<u8> + Default,
        AllocU16: Allocator<u16> + Default,
        AllocI32: Allocator<i32> + Default,
        AllocU32: Allocator<u32> + Default,
        AllocU64: Allocator<u64> + Default,
        AllocCommand: Allocator<Command> + Default,
        AllocFloatX: Allocator<floatX> + Default,
        AllocV8: Allocator<v8> + Default,
        AllocS16: Allocator<s16> + Default,
        AllocPDF: Allocator<PDF> + Default,
        AllocStaticCommand: Allocator<StaticCommand> + Default,
        AllocHistogramLiteral: Allocator<HistogramLiteral> + Default,
        AllocHistogramCommand: Allocator<HistogramCommand> + Default,
        AllocHistogramDistance: Allocator<HistogramDistance> + Default,
        AllocHistogramPair: Allocator<HistogramPair> + Default,
        AllocContextType: Allocator<ContextType> + Default,
        AllocHuffmanTree: Allocator<HuffmanTree> + Default,
        AllocZopfliNode: Allocator<ZopfliNode> + Default,
    > Default
    for CombiningAllocator<
        AllocU8,
        AllocU16,
        AllocI32,
        AllocU32,
        AllocU64,
        AllocCommand,
        AllocFloatX,
        AllocV8,
        AllocS16,
        AllocPDF,
        AllocStaticCommand,
        AllocHistogramLiteral,
        AllocHistogramCommand,
        AllocHistogramDistance,
        AllocHistogramPair,
        AllocContextType,
        AllocHuffmanTree,
        AllocZopfliNode,
    >
{
    fn default() -> Self {
        CombiningAllocator {
            alloc_u8: AllocU8::default(),
            alloc_u16: AllocU16::default(),
            alloc_i32: AllocI32::default(),
            alloc_u32: AllocU32::default(),
            alloc_u64: AllocU64::default(),
            alloc_c: AllocCommand::default(),
            alloc_f: AllocFloatX::default(),
            alloc_f32x8: AllocV8::default(),
            alloc_i16x16: AllocS16::default(),
            alloc_pdf: AllocPDF::default(),
            alloc_sc: AllocStaticCommand::default(),
            alloc_hl: AllocHistogramLiteral::default(),
            alloc_hc: AllocHistogramCommand::default(),
            alloc_hd: AllocHistogramDistance::default(),
            alloc_hp: AllocHistogramPair::default(),
            alloc_ct: AllocContextType::default(),
            alloc_ht: AllocHuffmanTree::default(),
            alloc_zn: AllocZopfliNode::default(),
        }
    }
}

impl<
        AllocU8: Allocator<u8> + Clone,
        AllocU16: Allocator<u16> + Clone,
        AllocI32: Allocator<i32> + Clone,
        AllocU32: Allocator<u32> + Clone,
        AllocU64: Allocator<u64> + Clone,
        AllocCommand: Allocator<Command> + Clone,
        AllocFloatX: Allocator<floatX> + Clone,
        AllocV8: Allocator<v8> + Clone,
        AllocS16: Allocator<s16> + Clone,
        AllocPDF: Allocator<PDF> + Clone,
        AllocStaticCommand: Allocator<StaticCommand> + Clone,
        AllocHistogramLiteral: Allocator<HistogramLiteral> + Clone,
        AllocHistogramCommand: Allocator<HistogramCommand> + Clone,
        AllocHistogramDistance: Allocator<HistogramDistance> + Clone,
        AllocHistogramPair: Allocator<HistogramPair> + Clone,
        AllocContextType: Allocator<ContextType> + Clone,
        AllocHuffmanTree: Allocator<HuffmanTree> + Clone,
        AllocZopfliNode: Allocator<ZopfliNode> + Clone,
    > Clone
    for CombiningAllocator<
        AllocU8,
        AllocU16,
        AllocI32,
        AllocU32,
        AllocU64,
        AllocCommand,
        AllocFloatX,
        AllocV8,
        AllocS16,
        AllocPDF,
        AllocStaticCommand,
        AllocHistogramLiteral,
        AllocHistogramCommand,
        AllocHistogramDistance,
        AllocHistogramPair,
        AllocContextType,
        AllocHuffmanTree,
        AllocZopfliNode,
    >
{
    fn clone(&self) -> Self {
        CombiningAllocator {
            alloc_u8: self.alloc_u8.clone(),
            alloc_u16: self.alloc_u16.clone(),
            alloc_i32: self.alloc_i32.clone(),
            alloc_u32: self.alloc_u32.clone(),
            alloc_u64: self.alloc_u64.clone(),
            alloc_c: self.alloc_c.clone(),
            alloc_f: self.alloc_f.clone(),
            alloc_f32x8: self.alloc_f32x8.clone(),
            alloc_i16x16: self.alloc_i16x16.clone(),
            alloc_pdf: self.alloc_pdf.clone(),
            alloc_sc: self.alloc_sc.clone(),
            alloc_hl: self.alloc_hl.clone(),
            alloc_hc: self.alloc_hc.clone(),
            alloc_hd: self.alloc_hd.clone(),
            alloc_hp: self.alloc_hp.clone(),
            alloc_ct: self.alloc_ct.clone(),
            alloc_ht: self.alloc_ht.clone(),
            alloc_zn: self.alloc_zn.clone(),
        }
    }
}

impl<
        AllocU8: Allocator<u8> + Copy,
        AllocU16: Allocator<u16> + Copy,
        AllocI32: Allocator<i32> + Copy,
        AllocU32: Allocator<u32> + Copy,
        AllocU64: Allocator<u64> + Copy,
        AllocCommand: Allocator<Command> + Copy,
        AllocFloatX: Allocator<floatX> + Copy,
        AllocV8: Allocator<v8> + Copy,
        AllocS16: Allocator<s16> + Copy,
        AllocPDF: Allocator<PDF> + Copy,
        AllocStaticCommand: Allocator<StaticCommand> + Copy,
        AllocHistogramLiteral: Allocator<HistogramLiteral> + Copy,
        AllocHistogramCommand: Allocator<HistogramCommand> + Copy,
        AllocHistogramDistance: Allocator<HistogramDistance> + Copy,
        AllocHistogramPair: Allocator<HistogramPair> + Copy,
        AllocContextType: Allocator<ContextType> + Copy,
        AllocHuffmanTree: Allocator<HuffmanTree> + Copy,
        AllocZopfliNode: Allocator<ZopfliNode> + Copy,
    > Copy
    for CombiningAllocator<
        AllocU8,
        AllocU16,
        AllocI32,
        AllocU32,
        AllocU64,
        AllocCommand,
        AllocFloatX,
        AllocV8,
        AllocS16,
        AllocPDF,
        AllocStaticCommand,
        AllocHistogramLiteral,
        AllocHistogramCommand,
        AllocHistogramDistance,
        AllocHistogramPair,
        AllocContextType,
        AllocHuffmanTree,
        AllocZopfliNode,
    >
{
}

macro_rules! implement_allocator {
    ($bound_name: ty,
   $type_name: ty,
   $sub_type_name: ty,
   $local_name: ident) => {
        impl<
                AllocU8: Allocator<u8>,
                AllocU16: Allocator<u16>,
                AllocI32: Allocator<i32>,
                AllocU32: Allocator<u32>,
                AllocU64: Allocator<u64>,
                AllocCommand: Allocator<Command>,
                AllocFloatX: Allocator<floatX>,
                AllocV8: Allocator<v8>,
                AllocS16: Allocator<s16>,
                AllocPDF: Allocator<PDF>,
                AllocStaticCommand: Allocator<StaticCommand>,
                AllocHistogramLiteral: Allocator<HistogramLiteral>,
                AllocHistogramCommand: Allocator<HistogramCommand>,
                AllocHistogramDistance: Allocator<HistogramDistance>,
                AllocHistogramPair: Allocator<HistogramPair>,
                AllocContextType: Allocator<ContextType>,
                AllocHuffmanTree: Allocator<HuffmanTree>,
                AllocZopfliNode: Allocator<ZopfliNode>,
            > Allocator<$type_name>
            for CombiningAllocator<
                AllocU8,
                AllocU16,
                AllocI32,
                AllocU32,
                AllocU64,
                AllocCommand,
                AllocFloatX,
                AllocV8,
                AllocS16,
                AllocPDF,
                AllocStaticCommand,
                AllocHistogramLiteral,
                AllocHistogramCommand,
                AllocHistogramDistance,
                AllocHistogramPair,
                AllocContextType,
                AllocHuffmanTree,
                AllocZopfliNode,
            >
        {
            type AllocatedMemory = $sub_type_name;
            fn alloc_cell(
                &mut self,
                size: usize,
            ) -> <Self as Allocator<$type_name>>::AllocatedMemory {
                self.$local_name.alloc_cell(size)
            }
            fn free_cell(&mut self, data: <Self as Allocator<$type_name>>::AllocatedMemory) {
                self.$local_name.free_cell(data)
            }
        }
    };
}

implement_allocator!(AllocU8, u8, AllocU8::AllocatedMemory, alloc_u8);
implement_allocator!(AllocU16, u16, AllocU16::AllocatedMemory, alloc_u16);

implement_allocator!(AllocI32, i32, AllocI32::AllocatedMemory, alloc_i32);
implement_allocator!(AllocU32, u32, AllocU32::AllocatedMemory, alloc_u32);
implement_allocator!(AllocU64, u64, AllocU64::AllocatedMemory, alloc_u64);
implement_allocator!(
    AllocCommand,
    Command,
    AllocCommand::AllocatedMemory,
    alloc_c
);
implement_allocator!(AllocFloatX, floatX, AllocFloatX::AllocatedMemory, alloc_f);
implement_allocator!(AllocV8, v8, AllocV8::AllocatedMemory, alloc_f32x8);
implement_allocator!(AllocS16, s16, AllocS16::AllocatedMemory, alloc_i16x16);
implement_allocator!(AllocPDF, PDF, AllocPDF::AllocatedMemory, alloc_pdf);
implement_allocator!(
    AllocStaticCommand,
    StaticCommand,
    AllocStaticCommand::AllocatedMemory,
    alloc_sc
);
implement_allocator!(
    AllocHistogramLiteral,
    HistogramLiteral,
    AllocHistogramLiteral::AllocatedMemory,
    alloc_hl
);
implement_allocator!(
    AllocHistogramCommand,
    HistogramCommand,
    AllocHistogramCommand::AllocatedMemory,
    alloc_hc
);
implement_allocator!(
    AllocHistogramDistance,
    HistogramDistance,
    AllocHistogramDistance::AllocatedMemory,
    alloc_hd
);
implement_allocator!(
    AllocHistogramPair,
    HistogramPair,
    AllocHistogramPair::AllocatedMemory,
    alloc_hp
);
implement_allocator!(
    AllocContextType,
    ContextType,
    AllocContextType::AllocatedMemory,
    alloc_ct
);
implement_allocator!(
    AllocHuffmanTree,
    HuffmanTree,
    AllocHuffmanTree::AllocatedMemory,
    alloc_ht
);
implement_allocator!(
    AllocZopfliNode,
    ZopfliNode,
    AllocZopfliNode::AllocatedMemory,
    alloc_zn
);

/// Helper function to allocate memory with the given allocator (may crash if the `len` is 0).
pub(crate) fn allocate<T, A: Allocator<T>>(alloc: &mut A, len: usize) -> A::AllocatedMemory {
    A::alloc_cell(alloc, len)
}

/// Helper function to create an empty allocator object
pub(crate) fn alloc_default<T, A: Allocator<T>>() -> A::AllocatedMemory {
    A::AllocatedMemory::default()
}

/// Helper function to allocate memory or return a default value if the condition is false
pub(crate) fn alloc_if<T, A: Allocator<T>>(
    condition: bool,
    alloc: &mut A,
    len: usize,
) -> A::AllocatedMemory {
    if condition {
        allocate(alloc, len)
    } else {
        alloc_default::<T, A>()
    }
}

/// Helper function to allocate memory or return a default value when the size is 0.
pub(crate) fn alloc_or_default<T, A: Allocator<T>>(
    alloc: &mut A,
    len: usize,
) -> A::AllocatedMemory {
    alloc_if(len > 0, alloc, len)
}
