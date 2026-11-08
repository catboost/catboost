//! Data-structures / interpretation for extended leafs (>= 0x8000_0000)
use bitflags::bitflags;
use core::fmt::{self, Debug, Display, Formatter};
use core::mem::size_of;
use core::slice;
use core::str;

use crate::{
    get_bits, CpuIdReader, CpuIdResult, Vendor, EAX_EXTENDED_CPU_TOPOLOGY,
    EAX_PQOS_EXTENDED_FEATURES,
};

/// Extended Processor and Processor Feature Identifiers (LEAF=0x8000_0001)
///
/// # Platforms
/// ✅ AMD 🟡 Intel
pub struct ExtendedProcessorFeatureIdentifiers {
    vendor: Vendor,
    eax: u32,
    ebx: u32,
    ecx: ExtendedFunctionInfoEcx,
    edx: ExtendedFunctionInfoEdx,
}

impl ExtendedProcessorFeatureIdentifiers {
    pub(crate) fn new(vendor: Vendor, data: CpuIdResult) -> Self {
        Self {
            vendor,
            eax: data.eax,
            ebx: data.ebx,
            ecx: ExtendedFunctionInfoEcx::from_bits_truncate(data.ecx),
            edx: ExtendedFunctionInfoEdx::from_bits_truncate(data.edx),
        }
    }

    /// Extended Processor Signature.
    ///
    /// # AMD
    /// The value returned is the same as the value returned in EAX for LEAF=0x0000_0001
    /// (use `CpuId.get_feature_info` instead)
    ///
    /// # Intel
    /// Vague mention of "Extended Processor Signature", not clear what it's supposed to
    /// represent.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn extended_signature(&self) -> u32 {
        self.eax
    }

    /// Returns package type on AMD.
    ///
    /// Package type. If `(Family[7:0] >= 10h)`, this field is valid. If
    /// `(Family[7:0]<10h)`, this field is reserved
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn pkg_type(&self) -> u32 {
        get_bits(self.ebx, 28, 31)
    }

    /// Returns brand ID on AMD.
    ///
    /// This field, in conjunction with CPUID `LEAF=0x0000_0001_EBX[8BitBrandId]`, and used
    /// by firmware to generate the processor name string.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn brand_id(&self) -> u32 {
        get_bits(self.ebx, 0, 15)
    }

    /// Is LAHF/SAHF available in 64-bit mode?
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_lahf_sahf(&self) -> bool {
        self.ecx.contains(ExtendedFunctionInfoEcx::LAHF_SAHF)
    }

    /// Check support legacy cmp.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_cmp_legacy(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::CMP_LEGACY)
    }

    /// Secure virtual machine supported.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_svm(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::SVM)
    }

    /// Extended APIC space.
    ///
    /// This bit indicates the presence of extended APIC register space starting at offset
    /// 400h from the “APIC Base Address Register,” as specified in the BKDG.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_ext_apic_space(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::EXT_APIC_SPACE)
    }

    /// LOCK MOV CR0 means MOV CR8. See “MOV(CRn)” in APM3.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_alt_mov_cr8(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::ALTMOVCR8)
    }

    /// Is LZCNT available?
    ///
    /// # AMD
    /// It's called ABM (Advanced bit manipulation) on AMD and also adds support for
    /// some other instructions.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_lzcnt(&self) -> bool {
        self.ecx.contains(ExtendedFunctionInfoEcx::LZCNT)
    }

    /// XTRQ, INSERTQ, MOVNTSS, and MOVNTSD instruction support.
    ///
    /// See “EXTRQ”, “INSERTQ”,“MOVNTSS”, and “MOVNTSD” in APM4.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_sse4a(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::SSE4A)
    }

    /// Misaligned SSE mode. See “Misaligned Access Support Added for SSE Instructions” in
    /// APM1.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_misaligned_sse_mode(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::MISALIGNSSE)
    }

    /// Is PREFETCHW available?
    ///
    /// # AMD
    /// PREFETCH and PREFETCHW instruction support.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_prefetchw(&self) -> bool {
        self.ecx.contains(ExtendedFunctionInfoEcx::PREFETCHW)
    }

    /// Indicates OS-visible workaround support
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_osvw(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::OSVW)
    }

    /// Instruction based sampling.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_ibs(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::IBS)
    }

    /// Extended operation support.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_xop(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::XOP)
    }

    /// SKINIT and STGI are supported.
    ///
    /// Indicates support for SKINIT and STGI, independent of the value of
    /// `MSRC000_0080[SVME]`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_skinit(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::SKINIT)
    }

    /// Watchdog timer support.
    ///
    /// Indicates support for MSRC001_0074.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_wdt(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::WDT)
    }

    /// Lightweight profiling support
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_lwp(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::LWP)
    }

    /// Four-operand FMA instruction support.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_fma4(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::FMA4)
    }

    /// Trailing bit manipulation instruction support.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_tbm(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::TBM)
    }

    /// Topology extensions support.
    ///
    /// Indicates support for CPUID `Fn8000_001D_EAX_x[N:0]-CPUID Fn8000_001E_EDX`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_topology_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::TOPEXT)
    }

    /// Processor performance counter extensions support.
    ///
    /// Indicates support for `MSRC001_020[A,8,6,4,2,0]` and `MSRC001_020[B,9,7,5,3,1]`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_perf_cntr_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::PERFCTREXT)
    }

    /// NB performance counter extensions support.
    ///
    /// Indicates support for `MSRC001_024[6,4,2,0]` and `MSRC001_024[7,5,3,1]`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_nb_perf_cntr_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::PERFCTREXTNB)
    }

    /// Data access breakpoint extension.
    ///
    /// Indicates support for `MSRC001_1027` and `MSRC001_101[B:9]`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_data_access_bkpt_extension(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::DATABRKPEXT)
    }

    /// Performance time-stamp counter.
    ///
    /// Indicates support for `MSRC001_0280` `[Performance Time Stamp Counter]`.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_perf_tsc(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::PERFTSC)
    }

    /// Support for L3 performance counter extension.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_perf_cntr_llc_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::PERFCTREXTLLC)
    }

    /// Support for MWAITX and MONITORX instructions.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_monitorx_mwaitx(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::MONITORX)
    }

    /// Breakpoint Addressing masking extended to bit 31.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_addr_mask_extension(&self) -> bool {
        self.vendor == Vendor::Amd && self.ecx.contains(ExtendedFunctionInfoEcx::ADDRMASKEXT)
    }

    /// Are fast system calls available.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_syscall_sysret(&self) -> bool {
        self.edx.contains(ExtendedFunctionInfoEdx::SYSCALL_SYSRET)
    }

    /// Is there support for execute disable bit.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_execute_disable(&self) -> bool {
        self.edx.contains(ExtendedFunctionInfoEdx::EXECUTE_DISABLE)
    }

    /// AMD extensions to MMX instructions.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_mmx_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.edx.contains(ExtendedFunctionInfoEdx::MMXEXT)
    }

    /// FXSAVE and FXRSTOR instruction optimizations.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_fast_fxsave_fxstor(&self) -> bool {
        self.vendor == Vendor::Amd && self.edx.contains(ExtendedFunctionInfoEdx::FFXSR)
    }

    /// Is there support for 1GiB pages.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_1gib_pages(&self) -> bool {
        self.edx.contains(ExtendedFunctionInfoEdx::GIB_PAGES)
    }

    /// Check support for rdtscp instruction.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_rdtscp(&self) -> bool {
        self.edx.contains(ExtendedFunctionInfoEdx::RDTSCP)
    }

    /// Check support for 64-bit mode.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_64bit_mode(&self) -> bool {
        self.edx.contains(ExtendedFunctionInfoEdx::I64BIT_MODE)
    }

    /// 3DNow AMD extensions.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_amd_3dnow_extensions(&self) -> bool {
        self.vendor == Vendor::Amd && self.edx.contains(ExtendedFunctionInfoEdx::THREEDNOWEXT)
    }

    /// 3DNow extensions.
    ///
    /// # Platform
    /// ✅ AMD ❌ Intel (will return false)
    pub fn has_3dnow(&self) -> bool {
        self.vendor == Vendor::Amd && self.edx.contains(ExtendedFunctionInfoEdx::THREEDNOW)
    }
}

impl Debug for ExtendedProcessorFeatureIdentifiers {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut ds = f.debug_struct("ExtendedProcessorFeatureIdentifiers");
        ds.field("extended_signature", &self.extended_signature());

        if self.vendor == Vendor::Amd {
            ds.field("pkg_type", &self.pkg_type());
            ds.field("brand_id", &self.brand_id());
        }
        ds.field("ecx_features", &self.ecx);
        ds.field("edx_features", &self.edx);
        ds.finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFunctionInfoEcx: u32 {
        const LAHF_SAHF = 1 << 0;
        const CMP_LEGACY =  1 << 1;
        const SVM = 1 << 2;
        const EXT_APIC_SPACE = 1 << 3;
        const ALTMOVCR8 = 1 << 4;
        const LZCNT = 1 << 5;
        const SSE4A = 1 << 6;
        const MISALIGNSSE = 1 << 7;
        const PREFETCHW = 1 << 8;
        const OSVW = 1 << 9;
        const IBS = 1 << 10;
        const XOP = 1 << 11;
        const SKINIT = 1 << 12;
        const WDT = 1 << 13;
        const LWP = 1 << 15;
        const FMA4 = 1 << 16;
        const TBM = 1 << 21;
        const TOPEXT = 1 << 22;
        const PERFCTREXT = 1 << 23;
        const PERFCTREXTNB = 1 << 24;
        const DATABRKPEXT = 1 << 26;
        const PERFTSC = 1 << 27;
        const PERFCTREXTLLC = 1 << 28;
        const MONITORX = 1 << 29;
        const ADDRMASKEXT = 1 << 30;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFunctionInfoEdx: u32 {
        const SYSCALL_SYSRET = 1 << 11;
        const EXECUTE_DISABLE = 1 << 20;
        const MMXEXT = 1 << 22;
        const FFXSR = 1 << 25;
        const GIB_PAGES = 1 << 26;
        const RDTSCP = 1 << 27;
        const I64BIT_MODE = 1 << 29;
        const THREEDNOWEXT = 1 << 30;
        const THREEDNOW = 1 << 31;
    }
}

/// Processor name (LEAF=0x8000_0002..=0x8000_0004).
///
/// ASCII string up to 48 characters in length corresponding to the processor name.
///
/// # Platforms
/// ✅ AMD ✅ Intel
pub struct ProcessorBrandString {
    data: [CpuIdResult; 3],
}

impl ProcessorBrandString {
    pub(crate) fn new(data: [CpuIdResult; 3]) -> Self {
        Self { data }
    }

    /// Return the processor brand string as a rust string.
    ///
    /// For example:
    /// "11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz".
    pub fn as_str(&self) -> &str {
        // Safety: CpuIdResult is laid out with repr(C), and the array
        // self.data contains 3 contiguous elements.
        let slice: &[u8] = unsafe {
            slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * size_of::<CpuIdResult>(),
            )
        };

        // Brand terminated at nul byte or end, whichever comes first.
        let slice = slice.split(|&x| x == 0).next().unwrap();
        str::from_utf8(slice)
            .unwrap_or("Invalid Processor Brand String")
            .trim()
    }
}

impl Debug for ProcessorBrandString {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProcessorBrandString")
            .field("as_str", &self.as_str())
            .finish()
    }
}

/// L1 Cache and TLB Information (LEAF=0x8000_0005).
///
/// # Availability
/// ✅ AMD ❌ Intel (reserved=0)
#[derive(PartialEq, Eq, Debug)]
pub struct L1CacheTlbInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl L1CacheTlbInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// Data TLB associativity for 2-MB and 4-MB pages.
    pub fn dtlb_2m_4m_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 24, 31) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// Data TLB number of entries for 2-MB and 4-MB pages.
    ///
    /// The value returned is for the number of entries available for the 2-MB page size;
    /// 4-MB pages require two 2-MB entries, so the number of entries available for the
    /// 4-MB page size is one-half the returned value.
    pub fn dtlb_2m_4m_size(&self) -> u8 {
        get_bits(self.eax, 16, 23) as u8
    }

    /// Instruction TLB associativity for 2-MB and 4-MB pages.
    pub fn itlb_2m_4m_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 8, 15) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// Instruction TLB number of entries for 2-MB and 4-MB pages.
    ///
    /// The value returned is for the number of entries available for the 2-MB page size;
    /// 4-MB pages require two 2-MB entries, so the number of entries available for the
    /// 4-MB page size is one-half the returned value.
    pub fn itlb_2m_4m_size(&self) -> u8 {
        get_bits(self.eax, 0, 7) as u8
    }

    /// Data TLB associativity for 4K pages.
    pub fn dtlb_4k_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 24, 31) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// Data TLB number of entries for 4K pages.
    pub fn dtlb_4k_size(&self) -> u8 {
        get_bits(self.ebx, 16, 23) as u8
    }

    /// Instruction TLB associativity for 4K pages.
    pub fn itlb_4k_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 8, 15) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// Instruction TLB number of entries for 4K pages.
    pub fn itlb_4k_size(&self) -> u8 {
        get_bits(self.ebx, 0, 7) as u8
    }

    /// L1 data cache size in KB
    pub fn dcache_size(&self) -> u8 {
        get_bits(self.ecx, 24, 31) as u8
    }

    /// L1 data cache associativity.
    pub fn dcache_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ecx, 16, 23) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// L1 data cache lines per tag.
    pub fn dcache_lines_per_tag(&self) -> u8 {
        get_bits(self.ecx, 8, 15) as u8
    }

    /// L1 data cache line size in bytes.
    pub fn dcache_line_size(&self) -> u8 {
        get_bits(self.ecx, 0, 7) as u8
    }

    /// L1 instruction cache size in KB
    pub fn icache_size(&self) -> u8 {
        get_bits(self.edx, 24, 31) as u8
    }

    /// L1 instruction cache associativity.
    pub fn icache_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.edx, 16, 23) as u8;
        Associativity::for_l1(assoc_bits)
    }

    /// L1 instruction cache lines per tag.
    pub fn icache_lines_per_tag(&self) -> u8 {
        get_bits(self.edx, 8, 15) as u8
    }

    /// L1 instruction cache line size in bytes.
    pub fn icache_line_size(&self) -> u8 {
        get_bits(self.edx, 0, 7) as u8
    }
}

/// L2/L3 Cache and TLB Information (LEAF=0x8000_0006).
///
/// # Availability
/// ✅ AMD 🟡 Intel
#[derive(PartialEq, Eq, Debug)]
pub struct L2And3CacheTlbInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl L2And3CacheTlbInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// L2 Data TLB associativity for 2-MB and 4-MB pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn dtlb_2m_4m_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 28, 31) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Data TLB number of entries for 2-MB and 4-MB pages.
    ///
    /// The value returned is for the number of entries available for the 2-MB page size;
    /// 4-MB pages require two 2-MB entries, so the number of entries available for the
    /// 4-MB page size is one-half the returned value.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn dtlb_2m_4m_size(&self) -> u16 {
        get_bits(self.eax, 16, 27) as u16
    }

    /// L2 Instruction TLB associativity for 2-MB and 4-MB pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn itlb_2m_4m_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 12, 15) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Instruction TLB number of entries for 2-MB and 4-MB pages.
    ///
    /// The value returned is for the number of entries available for the 2-MB page size;
    /// 4-MB pages require two 2-MB entries, so the number of entries available for the
    /// 4-MB page size is one-half the returned value.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn itlb_2m_4m_size(&self) -> u16 {
        get_bits(self.eax, 0, 11) as u16
    }

    /// L2 Data TLB associativity for 4K pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn dtlb_4k_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 28, 31) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Data TLB number of entries for 4K pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn dtlb_4k_size(&self) -> u16 {
        get_bits(self.ebx, 16, 27) as u16
    }

    /// L2 Instruction TLB associativity for 4K pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn itlb_4k_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 12, 15) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Instruction TLB number of entries for 4K pages.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn itlb_4k_size(&self) -> u16 {
        get_bits(self.ebx, 0, 11) as u16
    }

    /// L2 Cache Line size in bytes
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn l2cache_line_size(&self) -> u8 {
        get_bits(self.ecx, 0, 7) as u8
    }

    /// L2 cache lines per tag.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn l2cache_lines_per_tag(&self) -> u8 {
        get_bits(self.ecx, 8, 11) as u8
    }

    /// L2 Associativity field
    ///
    /// # Availability
    /// ✅ AMD ✅ Intel
    pub fn l2cache_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ecx, 12, 15) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// Cache size in KB.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn l2cache_size(&self) -> u16 {
        get_bits(self.ecx, 16, 31) as u16
    }

    /// L2 Cache Line size in bytes
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn l3cache_line_size(&self) -> u8 {
        get_bits(self.edx, 0, 7) as u8
    }

    /// L2 cache lines per tag.
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn l3cache_lines_per_tag(&self) -> u8 {
        get_bits(self.edx, 8, 11) as u8
    }

    /// L2 Associativity field
    ///
    /// # Availability
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn l3cache_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.edx, 12, 15) as u8;
        Associativity::for_l3(assoc_bits)
    }

    /// Specifies the L3 cache size range
    ///
    /// `(L3Size[31:18] * 512KB) <= L3 cache size < ((L3Size[31:18]+1) * 512KB)`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn l3cache_size(&self) -> u16 {
        get_bits(self.edx, 18, 31) as u16
    }
}

/// Info about cache Associativity.
#[derive(PartialEq, Eq, Debug)]
pub enum Associativity {
    Disabled,
    DirectMapped,
    NWay(u8),
    FullyAssociative,
    Unknown,
}

impl Display for Associativity {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let s = match self {
            Associativity::Disabled => "Disabled",
            Associativity::DirectMapped => "Direct mapped",
            Associativity::NWay(n) => {
                return write!(f, "NWay({})", n);
            }
            Associativity::FullyAssociative => "Fully associative",
            Associativity::Unknown => "Unknown (check leaf 0x8000_001d)",
        };
        f.write_str(s)
    }
}

impl Associativity {
    /// Constructor for L1 Cache and TLB Associativity Field Encodings
    fn for_l1(n: u8) -> Associativity {
        match n {
            0x0 => Associativity::Disabled, // Intel only, AMD is reserved
            0x1 => Associativity::DirectMapped,
            0x2..=0xfe => Associativity::NWay(n),
            0xff => Associativity::FullyAssociative,
        }
    }

    /// Constructor for L2 Cache and TLB Associativity Field Encodings
    fn for_l2(n: u8) -> Associativity {
        match n {
            0x0 => Associativity::Disabled,
            0x1 => Associativity::DirectMapped,
            0x2 => Associativity::NWay(2),
            0x4 => Associativity::NWay(4),
            0x5 => Associativity::NWay(6), // Reserved on Intel
            0x6 => Associativity::NWay(8),
            // 0x7 => SDM states: "See CPUID leaf 04H, sub-leaf 2"
            0x8 => Associativity::NWay(16),
            0x9 => Associativity::Unknown, // Intel: Reserved, AMD: Value for all fields should be determined from Fn8000_001D
            0xa => Associativity::NWay(32),
            0xb => Associativity::NWay(48),
            0xc => Associativity::NWay(64),
            0xd => Associativity::NWay(96),
            0xe => Associativity::NWay(128),
            0xF => Associativity::FullyAssociative,
            _ => Associativity::Unknown,
        }
    }

    /// Constructor for L2 Cache and TLB Associativity Field Encodings
    fn for_l3(n: u8) -> Associativity {
        Associativity::for_l2(n)
    }
}

/// Processor Power Management and RAS Capabilities (LEAF=0x8000_0007).
///
/// # Platforms
/// ✅ AMD 🟡 Intel
#[derive(Debug, PartialEq, Eq)]
pub struct ApmInfo {
    /// Reserved on AMD and Intel.
    _eax: u32,
    ebx: RasCapabilities,
    ecx: u32,
    edx: ApmInfoEdx,
}

impl ApmInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            _eax: data.eax,
            ebx: RasCapabilities::from_bits_truncate(data.ebx),
            ecx: data.ecx,
            edx: ApmInfoEdx::from_bits_truncate(data.edx),
        }
    }

    /// Is MCA overflow recovery available?
    ///
    /// If set, indicates that MCA overflow conditions (`MCi_STATUS[Overflow]=1`)
    /// are not fatal; software may safely ignore such conditions. If clear, MCA
    /// overflow conditions require software to shut down the system.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_mca_overflow_recovery(&self) -> bool {
        self.ebx.contains(RasCapabilities::MCAOVFLRECOV)
    }

    /// Has Software uncorrectable error containment and recovery capability?
    ///
    /// The processor supports software containment of uncorrectable errors
    /// through context synchronizing data poisoning and deferred error
    /// interrupts.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_succor(&self) -> bool {
        self.ebx.contains(RasCapabilities::SUCCOR)
    }

    /// Has Hardware assert supported?
    ///
    /// Indicates support for `MSRC001_10[DF:C0]`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_hwa(&self) -> bool {
        self.ebx.contains(RasCapabilities::HWA)
    }

    /// Specifies the ratio of the compute unit power accumulator sample period
    /// to the TSC counter period.
    ///
    /// Returns a value of 0 if not applicable for the system.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn cpu_pwr_sample_time_ratio(&self) -> u32 {
        self.ecx
    }

    /// Is Temperature Sensor available?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_ts(&self) -> bool {
        self.edx.contains(ApmInfoEdx::TS)
    }

    /// Frequency ID control.
    ///
    /// # Note
    /// Function replaced by `has_hw_pstate`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_freq_id_ctrl(&self) -> bool {
        self.edx.contains(ApmInfoEdx::FID)
    }

    /// Voltage ID control.
    ///
    /// # Note
    /// Function replaced by `has_hw_pstate`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_volt_id_ctrl(&self) -> bool {
        self.edx.contains(ApmInfoEdx::VID)
    }

    /// Has THERMTRIP?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_thermtrip(&self) -> bool {
        self.edx.contains(ApmInfoEdx::TTP)
    }

    /// Hardware thermal control (HTC)?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_tm(&self) -> bool {
        self.edx.contains(ApmInfoEdx::TM)
    }

    /// Has 100 MHz multiplier Control?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_100mhz_steps(&self) -> bool {
        self.edx.contains(ApmInfoEdx::MHZSTEPS100)
    }

    /// Has Hardware P-state control?
    ///
    /// MSRC001_0061 [P-state Current Limit], MSRC001_0062 [P-state Control] and
    /// MSRC001_0063 [P-state Status] exist
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_hw_pstate(&self) -> bool {
        self.edx.contains(ApmInfoEdx::HWPSTATE)
    }

    /// Is Invariant TSC available?
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_invariant_tsc(&self) -> bool {
        self.edx.contains(ApmInfoEdx::INVTSC)
    }

    /// Has Core performance boost?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_cpb(&self) -> bool {
        self.edx.contains(ApmInfoEdx::CPB)
    }

    /// Has Read-only effective frequency interface?
    ///
    /// Indicates presence of MSRC000_00E7 [Read-Only Max Performance Frequency
    /// Clock Count (MPerfReadOnly)] and MSRC000_00E8 [Read-Only Actual
    /// Performance Frequency Clock Count (APerfReadOnly)].
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_ro_effective_freq_iface(&self) -> bool {
        self.edx.contains(ApmInfoEdx::EFFFREQRO)
    }

    /// Indicates support for processor feedback interface.
    ///
    /// # Note
    /// This feature is deprecated.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_feedback_iface(&self) -> bool {
        self.edx.contains(ApmInfoEdx::PROCFEEDBACKIF)
    }

    /// Has Processor power reporting interface?
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_power_reporting_iface(&self) -> bool {
        self.edx.contains(ApmInfoEdx::PROCPWRREPORT)
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ApmInfoEdx: u32 {
        const TS = 1 << 0;
        const FID = 1 << 1;
        const VID = 1 << 2;
        const TTP = 1 << 3;
        const TM = 1 << 4;
        const MHZSTEPS100 = 1 << 6;
        const HWPSTATE = 1 << 7;
        const INVTSC = 1 << 8;
        const CPB = 1 << 9;
        const EFFFREQRO = 1 << 10;
        const PROCFEEDBACKIF = 1 << 11;
        const PROCPWRREPORT = 1 << 12;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct RasCapabilities: u32 {
        const MCAOVFLRECOV = 1 << 0;
        const SUCCOR = 1 << 1;
        const HWA = 1 << 2;
    }
}

/// Processor Capacity Parameters and Extended Feature Identification
/// (LEAF=0x8000_0008).
///
/// This function provides the size or capacity of various architectural
/// parameters that vary by implementation, as well as an extension to the
/// 0x8000_0001 feature identifiers.
///
/// # Platforms
/// ✅ AMD 🟡 Intel
#[derive(PartialEq, Eq)]
pub struct ProcessorCapacityAndFeatureInfo {
    eax: u32,
    ebx: ProcessorCapacityAndFeatureEbx,
    ecx: u32,
    edx: u32,
}

impl ProcessorCapacityAndFeatureInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: ProcessorCapacityAndFeatureEbx::from_bits_truncate(data.ebx),
            ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// Physical Address Bits
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn physical_address_bits(&self) -> u8 {
        get_bits(self.eax, 0, 7) as u8
    }

    /// Linear Address Bits
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn linear_address_bits(&self) -> u8 {
        get_bits(self.eax, 8, 15) as u8
    }

    /// Guest Physical Address Bits
    ///
    /// This number applies only to guests using nested paging. When this field
    /// is zero, refer to the PhysAddrSize field for the maximum guest physical
    /// address size.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn guest_physical_address_bits(&self) -> u8 {
        get_bits(self.eax, 16, 23) as u8
    }

    /// CLZERO instruction supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_cl_zero(&self) -> bool {
        self.ebx.contains(ProcessorCapacityAndFeatureEbx::CLZERO)
    }

    /// Instruction Retired Counter MSR available if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_inst_ret_cntr_msr(&self) -> bool {
        self.ebx
            .contains(ProcessorCapacityAndFeatureEbx::INST_RETCNT_MSR)
    }

    /// FP Error Pointers Restored by XRSTOR if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_restore_fp_error_ptrs(&self) -> bool {
        self.ebx
            .contains(ProcessorCapacityAndFeatureEbx::RSTR_FP_ERR_PTRS)
    }

    /// INVLPGB and TLBSYNC instruction supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_invlpgb(&self) -> bool {
        self.ebx.contains(ProcessorCapacityAndFeatureEbx::INVLPGB)
    }

    /// RDPRU instruction supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_rdpru(&self) -> bool {
        self.ebx.contains(ProcessorCapacityAndFeatureEbx::RDPRU)
    }

    /// MCOMMIT instruction supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_mcommit(&self) -> bool {
        self.ebx.contains(ProcessorCapacityAndFeatureEbx::MCOMMIT)
    }

    /// WBNOINVD instruction supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_wbnoinvd(&self) -> bool {
        self.ebx.contains(ProcessorCapacityAndFeatureEbx::WBNOINVD)
    }

    /// WBINVD/WBNOINVD are interruptible if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_int_wbinvd(&self) -> bool {
        self.ebx
            .contains(ProcessorCapacityAndFeatureEbx::INT_WBINVD)
    }

    /// EFER.LMSLE is unsupported if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_unsupported_efer_lmsle(&self) -> bool {
        self.ebx
            .contains(ProcessorCapacityAndFeatureEbx::EFER_LMSLE_UNSUPP)
    }

    /// INVLPGB support for invalidating guest nested translations if set.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn has_invlpgb_nested(&self) -> bool {
        self.ebx
            .contains(ProcessorCapacityAndFeatureEbx::INVLPGB_NESTED)
    }

    /// Performance time-stamp counter size (in bits).
    ///
    /// Indicates the size of `MSRC001_0280[PTSC]`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=false)
    pub fn perf_tsc_size(&self) -> usize {
        let s = get_bits(self.ecx, 16, 17) as u8;
        match s & 0b11 {
            0b00 => 40,
            0b01 => 48,
            0b10 => 56,
            0b11 => 64,
            _ => unreachable!("AND with 0b11 in match"),
        }
    }

    /// APIC ID size.
    ///
    /// A value of zero indicates that legacy methods must be used to determine
    /// the maximum number of logical processors, as indicated by CPUID
    /// `Fn8000_0008_ECX[NC]`.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn apic_id_size(&self) -> u8 {
        get_bits(self.ecx, 12, 15) as u8
    }

    /// The size of the `apic_id_size` field determines the maximum number of
    /// logical processors (MNLP) that the package could theoretically support,
    /// and not the actual number of logical processors that are implemented or
    /// enabled in the package, as indicated by CPUID `Fn8000_0008_ECX[NC]`.
    ///
    /// `MNLP = (2 raised to the power of ApicIdSize[3:0])` (if not 0)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn maximum_logical_processors(&self) -> usize {
        usize::pow(2, self.apic_id_size() as u32)
    }

    /// Number of physical threads in the processor.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn num_phys_threads(&self) -> usize {
        get_bits(self.ecx, 0, 7) as usize + 1
    }

    /// Maximum page count for INVLPGB instruction.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn invlpgb_max_pages(&self) -> u16 {
        get_bits(self.edx, 0, 15) as u16
    }

    /// The maximum ECX value recognized by RDPRU.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved=0)
    pub fn max_rdpru_id(&self) -> u16 {
        get_bits(self.edx, 16, 31) as u16
    }
}

impl Debug for ProcessorCapacityAndFeatureInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessorCapacityAndFeatureInfo")
            .field("physical_address_bits", &self.physical_address_bits())
            .field("linear_address_bits", &self.linear_address_bits())
            .field(
                "guest_physical_address_bits",
                &self.guest_physical_address_bits(),
            )
            .field("has_cl_zero", &self.has_cl_zero())
            .field("has_inst_ret_cntr_msr", &self.has_inst_ret_cntr_msr())
            .field(
                "has_restore_fp_error_ptrs",
                &self.has_restore_fp_error_ptrs(),
            )
            .field("has_invlpgb", &self.has_invlpgb())
            .field("has_rdpru", &self.has_rdpru())
            .field("has_mcommit", &self.has_mcommit())
            .field("has_wbnoinvd", &self.has_wbnoinvd())
            .field("has_int_wbinvd", &self.has_int_wbinvd())
            .field(
                "has_unsupported_efer_lmsle",
                &self.has_unsupported_efer_lmsle(),
            )
            .field("has_invlpgb_nested", &self.has_invlpgb_nested())
            .field("perf_tsc_size", &self.perf_tsc_size())
            .field("apic_id_size", &self.apic_id_size())
            .field(
                "maximum_logical_processors",
                &self.maximum_logical_processors(),
            )
            .field("num_phys_threads", &self.num_phys_threads())
            .field("invlpgb_max_pages", &self.invlpgb_max_pages())
            .field("max_rdpru_id", &self.max_rdpru_id())
            .finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProcessorCapacityAndFeatureEbx: u32 {
        const CLZERO = 1 << 0;
        const INST_RETCNT_MSR = 1 << 1;
        const RSTR_FP_ERR_PTRS = 1 << 2;
        const INVLPGB = 1 << 3;
        const RDPRU = 1 << 4;
        const MCOMMIT = 1 << 8;
        const WBNOINVD = 1 << 9;
        const INT_WBINVD = 1 << 13;
        const EFER_LMSLE_UNSUPP = 1 << 20;
        const INVLPGB_NESTED = 1 << 21;
    }
}

/// Information about the SVM features that the processory supports (LEAF=0x8000_000A).
///
/// # Note
/// If SVM is not supported ([ExtendedProcessorFeatureIdentifiers::has_svm] is false),
/// this leaf is reserved ([crate::CpuId] will return None in this case).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct SvmFeatures {
    eax: u32,
    ebx: u32,
    /// Reserved
    _ecx: u32,
    edx: SvmFeaturesEdx,
}

impl SvmFeatures {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            _ecx: data.ecx,
            edx: SvmFeaturesEdx::from_bits_truncate(data.edx),
        }
    }

    /// SVM revision number.
    pub fn revision(&self) -> u8 {
        get_bits(self.eax, 0, 7) as u8
    }

    /// Number of available address space identifiers (ASID).
    pub fn supported_asids(&self) -> u32 {
        self.ebx
    }

    /// Nested paging supported if set.
    pub fn has_nested_paging(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::NP)
    }

    /// Indicates support for LBR Virtualization.
    pub fn has_lbr_virtualization(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::LBR_VIRT)
    }

    /// Indicates support for SVM-Lock if set.
    pub fn has_svm_lock(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::SVML)
    }

    /// Indicates support for NRIP save on #VMEXIT if set.
    pub fn has_nrip(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::NRIPS)
    }

    /// Indicates support for MSR TSC ratio (MSR `0xC000_0104`) if set.
    pub fn has_tsc_rate_msr(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::TSC_RATE_MSR)
    }

    /// Indicates support for VMCB clean bits if set.
    pub fn has_vmcb_clean_bits(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::VMCB_CLEAN)
    }

    /// Indicates that TLB flush events, including CR3 writes and CR4.PGE toggles, flush
    /// only the current ASID's TLB entries.
    ///
    /// Also indicates support for the extended VMCB TLB_Control.
    pub fn has_flush_by_asid(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::FLUSH_BY_ASID)
    }

    /// Indicates support for the decode assists if set.
    pub fn has_decode_assists(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::DECODE_ASSISTS)
    }

    /// Indicates support for the pause intercept filter if set.
    pub fn has_pause_filter(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::PAUSE_FILTER)
    }

    /// Indicates support for the PAUSE filter cycle count threshold if set.
    pub fn has_pause_filter_threshold(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::PAUSE_FILTER_THRESHOLD)
    }

    /// Support for the AMD advanced virtual interrupt controller if set.
    pub fn has_avic(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::AVIC)
    }

    /// VMSAVE and VMLOAD virtualization supported if set.
    pub fn has_vmsave_virtualization(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::VMSAVE_VIRT)
    }

    /// GIF -- virtualized global interrupt flag if set.
    pub fn has_gif(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::VGIF)
    }

    /// Guest Mode Execution Trap supported if set.
    pub fn has_gmet(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::GMET)
    }

    /// SVM supervisor shadow stack restrictions if set.
    pub fn has_sss_check(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::SSS_CHECK)
    }

    /// SPEC_CTRL virtualization supported if set.
    pub fn has_spec_ctrl(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::SPEC_CTRL)
    }

    /// When host `CR4.MCE=1` and guest `CR4.MCE=0`, machine check exceptions (`#MC`) in a
    /// guest do not cause shutdown and are always intercepted if set.
    pub fn has_host_mce_override(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::HOST_MCE_OVERRIDE)
    }

    /// Support for INVLPGB/TLBSYNC hypervisor enable in VMCB and TLBSYNC intercept if
    /// set.
    pub fn has_tlb_ctrl(&self) -> bool {
        self.edx.contains(SvmFeaturesEdx::TLB_CTL)
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SvmFeaturesEdx: u32 {
        const NP = 1 << 0;
        const LBR_VIRT = 1 << 1;
        const SVML = 1 << 2;
        const NRIPS = 1 << 3;
        const TSC_RATE_MSR = 1 << 4;
        const VMCB_CLEAN = 1 << 5;
        const FLUSH_BY_ASID = 1 << 6;
        const DECODE_ASSISTS = 1 << 7;
        const PAUSE_FILTER = 1 << 10;
        const PAUSE_FILTER_THRESHOLD = 1 << 12;
        const AVIC = 1 << 13;
        const VMSAVE_VIRT = 1 << 15;
        const VGIF = 1 << 16;
        const GMET = 1 << 17;
        const SSS_CHECK = 1 << 19;
        const SPEC_CTRL = 1 << 20;
        const HOST_MCE_OVERRIDE = 1 << 23;
        const TLB_CTL = 1 << 24;
    }
}

/// TLB 1-GiB Pages Information (LEAF=0x8000_0019).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct Tlb1gbPageInfo {
    eax: u32,
    ebx: u32,
    /// Reserved
    _ecx: u32,
    /// Reserved
    _edx: u32,
}

impl Tlb1gbPageInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// L1 Data TLB associativity for 1-GB pages.
    pub fn dtlb_l1_1gb_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 28, 31) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L1 Data TLB number of entries for 1-GB pages.
    pub fn dtlb_l1_1gb_size(&self) -> u8 {
        get_bits(self.eax, 16, 27) as u8
    }

    /// L1 Instruction TLB associativity for 1-GB pages.
    pub fn itlb_l1_1gb_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.eax, 12, 15) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L1 Instruction TLB number of entries for 1-GB pages.
    pub fn itlb_l1_1gb_size(&self) -> u8 {
        get_bits(self.eax, 0, 11) as u8
    }

    /// L2 Data TLB associativity for 1-GB pages.
    pub fn dtlb_l2_1gb_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 28, 31) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Data TLB number of entries for 1-GB pages.
    pub fn dtlb_l2_1gb_size(&self) -> u8 {
        get_bits(self.ebx, 16, 27) as u8
    }

    /// L2 Instruction TLB associativity for 1-GB pages.
    pub fn itlb_l2_1gb_associativity(&self) -> Associativity {
        let assoc_bits = get_bits(self.ebx, 12, 15) as u8;
        Associativity::for_l2(assoc_bits)
    }

    /// L2 Instruction TLB number of entries for 1-GB pages.
    pub fn itlb_l2_1gb_size(&self) -> u8 {
        get_bits(self.ebx, 0, 11) as u8
    }
}

/// Performance Optimization Identifier (LEAF=0x8000_001A).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct PerformanceOptimizationInfo {
    eax: PerformanceOptimizationInfoEax,
    /// Reserved
    _ebx: u32,
    /// Reserved
    _ecx: u32,
    /// Reserved
    _edx: u32,
}

impl PerformanceOptimizationInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: PerformanceOptimizationInfoEax::from_bits_truncate(data.eax),
            _ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// The internal FP/SIMD execution datapath is 128 bits wide if set.
    pub fn has_fp128(&self) -> bool {
        self.eax.contains(PerformanceOptimizationInfoEax::FP128)
    }

    /// MOVU (Move Unaligned) SSE instructions are efficient more than
    /// MOVL/MOVH SSE if set.
    pub fn has_movu(&self) -> bool {
        self.eax.contains(PerformanceOptimizationInfoEax::MOVU)
    }

    /// The internal FP/SIMD execution datapath is 256 bits wide if set.
    pub fn has_fp256(&self) -> bool {
        self.eax.contains(PerformanceOptimizationInfoEax::FP256)
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PerformanceOptimizationInfoEax: u32 {
        const FP128 = 1 << 0;
        const MOVU = 1 << 1;
        const FP256 = 1 << 2;
    }
}

/// Performance Optimization Identifier (LEAF=0x8000_001A).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct InstructionBasedSamplingCapabilities {
    eax: InstructionBasedSamplingCapabilitiesEax,
    /// Reserved
    _ebx: u32,
    /// Reserved
    _ecx: u32,
    /// Reserved
    _edx: u32,
}

impl InstructionBasedSamplingCapabilities {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: InstructionBasedSamplingCapabilitiesEax::from_bits_truncate(data.eax),
            _ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// IBS feature flags valid if set.
    pub fn has_feature_flags(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::IBSFFV)
    }

    /// IBS fetch sampling supported if set.
    pub fn has_fetch_sampling(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::FETCH_SAM)
    }

    /// IBS execution sampling supported if set.
    pub fn has_execution_sampling(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::OP_SAM)
    }

    /// Read write of op counter supported if set.
    pub fn has_read_write_operation_counter(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::RD_WR_OP_CNT)
    }

    /// Op counting mode supported if set.
    pub fn has_operation_counter(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::OP_CNT)
    }

    /// Branch target address reporting supported if set.
    pub fn has_branch_target_address_reporting(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::BRN_TRGT)
    }

    /// IbsOpCurCnt and IbsOpMaxCnt extend by 7 bits if set.
    pub fn has_operation_counter_extended(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::OP_CNT_EXT)
    }

    /// Invalid RIP indication supported if set.
    pub fn has_invalid_rip_indication(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::RIP_INVALID_CHK)
    }

    /// Fused branch micro-op indication supported if set.
    pub fn has_fused_branch_micro_op_indication(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::OP_BRN_FUSE)
    }

    /// L3 Miss Filtering for IBS supported if set.
    pub fn has_l3_miss_filtering(&self) -> bool {
        self.eax
            .contains(InstructionBasedSamplingCapabilitiesEax::IBS_L3_MISS_FILTERING)
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct InstructionBasedSamplingCapabilitiesEax: u32 {
        const IBSFFV = 1 << 0;
        const FETCH_SAM = 1 << 1;
        const OP_SAM = 1 << 2;
        const RD_WR_OP_CNT = 1 << 3;
        const OP_CNT = 1 << 4;
        const BRN_TRGT = 1 << 5;
        const OP_CNT_EXT = 1 << 6;
        const RIP_INVALID_CHK = 1 << 7;
        const OP_BRN_FUSE = 1 << 8;
        const IBS_L3_MISS_FILTERING = 1 << 11;
    }
}

/// Processor Topology Information (LEAF=0x8000_001E).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq)]
pub struct ProcessorTopologyInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    /// Reserved
    _edx: u32,
}

impl ProcessorTopologyInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// x2APIC ID
    pub fn x2apic_id(&self) -> u32 {
        self.eax
    }

    /// Core ID
    ///
    /// # Note
    /// `Core ID` means `Compute Unit ID` if AMD Family 15h-16h Processors.
    pub fn core_id(&self) -> u8 {
        get_bits(self.ebx, 0, 7) as u8
    }

    /// Threads per core
    ///
    /// # Note
    /// `Threads per Core` means `Cores per Compute Unit` if AMD Family 15h-16h Processors.
    pub fn threads_per_core(&self) -> u8 {
        get_bits(self.ebx, 8, 15) as u8 + 1
    }

    /// Node ID
    pub fn node_id(&self) -> u8 {
        get_bits(self.ecx, 0, 7) as u8
    }

    /// Nodes per processor
    pub fn nodes_per_processor(&self) -> u8 {
        get_bits(self.ecx, 8, 10) as u8 + 1
    }
}

impl Debug for ProcessorTopologyInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProcessorTopologyInfo")
            .field("x2apic_id", &self.x2apic_id())
            .field("core_id", &self.core_id())
            .field("threads_per_core", &self.threads_per_core())
            .field("node_id", &self.node_id())
            .field("nodes_per_processor", &self.nodes_per_processor())
            .finish()
    }
}

/// Encrypted Memory Capabilities (LEAF=0x8000_001F).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(Debug, PartialEq, Eq)]
pub struct MemoryEncryptionInfo {
    eax: MemoryEncryptionInfoEax,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl MemoryEncryptionInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: MemoryEncryptionInfoEax::from_bits_truncate(data.eax),
            ebx: data.ebx,
            ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// Secure Memory Encryption is supported if set.
    pub fn has_sme(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::SME)
    }

    /// Secure Encrypted Virtualization is supported if set.
    pub fn has_sev(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::SEV)
    }

    /// The Page Flush MSR is available if set.
    pub fn has_page_flush_msr(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::PAGE_FLUSH_MSR)
    }

    /// SEV Encrypted State is supported if set.
    pub fn has_sev_es(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::SEV_ES)
    }

    /// SEV Secure Nested Paging supported if set.
    pub fn has_sev_snp(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::SEV_SNP)
    }

    /// VM Permission Levels supported if set.
    pub fn has_vmpl(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::VMPL)
    }

    /// Hardware cache coherency across encryption domains enforced if set.
    pub fn has_hw_enforced_cache_coh(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::HWENFCACHECOH)
    }

    /// SEV guest execution only allowed from a 64-bit host if set.
    pub fn has_64bit_mode(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::HOST64)
    }

    /// Restricted Injection supported if set.
    pub fn has_restricted_injection(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::RESTINJECT)
    }

    /// Alternate Injection supported if set.
    pub fn has_alternate_injection(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::ALTINJECT)
    }

    /// Full debug state swap supported for SEV-ES guests.
    pub fn has_debug_swap(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::DBGSWP)
    }

    /// Disallowing IBS use by the host supported if set.
    pub fn has_prevent_host_ibs(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::PREVHOSTIBS)
    }

    /// Virtual Transparent Encryption supported if set.
    pub fn has_vte(&self) -> bool {
        self.eax.contains(MemoryEncryptionInfoEax::VTE)
    }

    /// C-bit location in page table entry
    pub fn c_bit_position(&self) -> u8 {
        get_bits(self.ebx, 0, 5) as u8
    }

    /// Physical Address bit reduction
    pub fn physical_address_reduction(&self) -> u8 {
        get_bits(self.ebx, 6, 11) as u8
    }

    /// Number of encrypted guests supported simultaneouslys
    pub fn max_encrypted_guests(&self) -> u32 {
        self.ecx
    }

    /// Minimum ASID value for an SEV enabled, SEV-ES disabled guest
    pub fn min_sev_no_es_asid(&self) -> u32 {
        self.edx
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MemoryEncryptionInfoEax: u32 {
        const SME = 1 << 0;
        const SEV = 1 << 1;
        const PAGE_FLUSH_MSR = 1 << 2;
        const SEV_ES = 1 << 3;
        const SEV_SNP = 1 << 4;
        const VMPL = 1 << 5;
        const HWENFCACHECOH = 1 << 10;
        const HOST64 = 1 << 11;
        const RESTINJECT = 1 << 12;
        const ALTINJECT = 1 << 13;
        const DBGSWP = 1 << 14;
        const PREVHOSTIBS = 1 << 15;
        const VTE = 1 << 16;
    }
}

/// Platform Quality of Service Information (LEAF=0x8000_0020).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq)]
pub struct PqosExtendedFeatureInfo<R: CpuIdReader> {
    read: R,
    _eax: u32,
    ebx: PqosExtendedFeatureInfoEbx,
    _ecx: u32,
    _edx: u32,
}

impl<R: CpuIdReader> PqosExtendedFeatureInfo<R> {
    pub(crate) fn new(read: R) -> Self {
        let data = read.cpuid2(EAX_PQOS_EXTENDED_FEATURES, 0);
        Self {
            read,
            _eax: data.eax,
            ebx: PqosExtendedFeatureInfoEbx::from_bits_truncate(data.ebx),
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// Memory Bandwidth Enforcement is supported if set.
    pub fn has_l3mbe(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::L3MBE)
    }

    /// Slow Memory Bandwidth Enforcement is supported if set.
    pub fn has_l3smbe(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::L3SMBE)
    }

    /// Bandwidth Monitoring Event Configuration is supported if set.
    pub fn has_bmec(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::BMEC)
    }

    /// L3 Range Reservations. See “L3 Range Reservation” in APM
    /// Volume 2 is supported if set.
    pub fn has_l3rr(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::L3RR)
    }

    /// Assignable Bandwidth Monitoring Counters is supported if set.
    pub fn has_abmc(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::ABMC)
    }

    /// Smart Data Cache Injection (SDCI) Allocation Enforcement is supported if set.
    pub fn has_sdciae(&self) -> bool {
        self.ebx.contains(PqosExtendedFeatureInfoEbx::SDCIAE)
    }

    /// Get L3 Memory Bandwidth Enforcement Information
    pub fn get_l3_memory_bandwidth_enforcement_info(
        &self,
    ) -> Option<L3MemoryBandwidthEnforcementInformation> {
        if self.has_l3mbe() {
            Some(L3MemoryBandwidthEnforcementInformation::new(
                self.read.cpuid2(EAX_PQOS_EXTENDED_FEATURES, 1),
            ))
        } else {
            None
        }
    }

    /// Get L3 Slow Memory Bandwidth Enforcement Information
    pub fn get_l3_slow_memory_bandwidth_enforcement_info(
        &self,
    ) -> Option<L3MemoryBandwidthEnforcementInformation> {
        if self.has_l3smbe() {
            Some(L3MemoryBandwidthEnforcementInformation::new(
                self.read.cpuid2(EAX_PQOS_EXTENDED_FEATURES, 2),
            ))
        } else {
            None
        }
    }

    /// Get Bandwidth Monitoring Event Counters Information
    pub fn get_bandwidth_monitoring_event_counters_info(
        &self,
    ) -> Option<BandwidthMonitoringEventCounters> {
        if self.has_bmec() {
            Some(BandwidthMonitoringEventCounters::new(
                self.read.cpuid2(EAX_PQOS_EXTENDED_FEATURES, 3),
            ))
        } else {
            None
        }
    }

    /// Get Bandwidth Monitoring Event Counters Information
    pub fn get_assignable_bandwidth_monitoring_counters_info(
        &self,
    ) -> Option<AssignableBandwidthMonitoringCounterInfo> {
        if self.has_abmc() {
            Some(AssignableBandwidthMonitoringCounterInfo::new(
                self.read.cpuid2(EAX_PQOS_EXTENDED_FEATURES, 5),
            ))
        } else {
            None
        }
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PqosExtendedFeatureInfoEbx: u32 {
        const L3MBE = 1 << 1;
        const L3SMBE = 1 << 2;
        const BMEC = 1 << 3;
        const L3RR = 1 << 4;
        const ABMC = 1 << 5;
        const SDCIAE = 1 << 6;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PqosExtendedFeatureInfoEbx5: u32 {
        const SELECT_COS = 1 << 0;
    }
}

impl<R: CpuIdReader> Debug for PqosExtendedFeatureInfo<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("PqosExtendedFeatureInfo")
            .field("has_l3mbe", &self.has_l3mbe())
            .field("has_l3smbe", &self.has_l3smbe())
            .field("has_bmec", &self.has_bmec())
            .field("has_l3rr", &self.has_l3rr())
            .field("has_abmc", &self.has_abmc())
            .field("has_sdciae", &self.has_sdciae())
            .finish()
    }
}

/// L3 Memory Bandwidth Enforcement Information (LEAF=0x8000_0020_x1 and x2).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct L3MemoryBandwidthEnforcementInformation {
    eax: u32,
    _ebx: u32,
    _ecx: u32,
    edx: u32,
}

impl L3MemoryBandwidthEnforcementInformation {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            _ebx: data.ebx,
            _ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// Identifies the size of the bandwidth specifier field in the
    /// L3QOS_BW_Control_n MSRs
    pub fn bandwidth_length(&self) -> u32 {
        self.eax
    }

    /// Maximum COS number supported by the L3MBE feature
    pub fn cos_max(&self) -> u32 {
        self.edx
    }
}

/// Bandwidth Monitoring Event Counters Information (LEAF=0x8000_0020_x3).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct BandwidthMonitoringEventCounters {
    _eax: u32,
    ebx: u32,
    ecx: BandwidthMonitoringEventCountersEcx,
    _edx: u32,
}

impl BandwidthMonitoringEventCounters {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            _eax: data.eax,
            ebx: data.ebx,
            ecx: BandwidthMonitoringEventCountersEcx::from_bits_truncate(data.ecx),
            _edx: data.edx,
        }
    }

    /// Get Number of configurable bandwidth events
    pub fn number_events(&self) -> u32 {
        get_bits(self.ebx, 0, 7)
    }

    /// Reads to local DRAM memory is supported if set.
    pub fn has_l3_cache_lcl_bw_fill_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_LCL_BW_FILL_MON)
    }

    /// Reads to remote DRAM memory is supported if set.
    pub fn has_l3_cache_rmt_bw_fill_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_RMT_BW_FILL_MON)
    }

    /// Non-temporal writes to local memory is supported if set.
    pub fn has_l3_cache_lcl_bw_nt_wr_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_LCL_BW_NT_WR_MON)
    }

    /// Non-temporal writes to remote memory is supported if set.
    pub fn has_l3_cache_rmt_bw_nt_wr_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_RMT_BW_NT_WR_MON)
    }

    /// Reads to local memory identified as “Slow Memory” is supported if set.
    pub fn has_l3_cache_lcl_slow_bw_fill_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_LCL_SLOW_BW_FILL_MON)
    }

    /// Reads to remote memory identified as “Slow Memory” is supported if set.
    pub fn has_l3_cache_rmt_slow_bw_fill_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_RMT_SLOW_BW_FILL_MON)
    }

    /// Dirty victim writes to all types of memory is supported if set.
    pub fn has_l3_cache_vic_mon(&self) -> bool {
        self.ecx
            .contains(BandwidthMonitoringEventCountersEcx::L3_CACHE_VIC_MON)
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BandwidthMonitoringEventCountersEcx: u32 {
        const L3_CACHE_LCL_BW_FILL_MON = 1 << 0;
        const L3_CACHE_RMT_BW_FILL_MON = 1 << 1;
        const L3_CACHE_LCL_BW_NT_WR_MON = 1 << 2;
        const L3_CACHE_RMT_BW_NT_WR_MON = 1 << 3;
        const L3_CACHE_LCL_SLOW_BW_FILL_MON = 1 << 4;
        const L3_CACHE_RMT_SLOW_BW_FILL_MON = 1 << 5;
        const L3_CACHE_VIC_MON = 1 << 6;
    }
}

/// L3 Memory Bandwidth Enforcement Information (LEAF=0x8000_0020_x5).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct AssignableBandwidthMonitoringCounterInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    _edx: u32,
}

impl AssignableBandwidthMonitoringCounterInfo {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// Get QM_CTR counter width, offset from 24 bits
    pub fn counter_size(&self) -> u8 {
        get_bits(self.eax, 0, 7) as u8
    }

    /// Indicates that QM_CTR bit 61 is an overflow bit if set
    pub fn has_overflow_bit(&self) -> bool {
        (self.eax & (1 << 8)) > 0
    }

    /// Get Maximum supported ABMC counter ID
    pub fn max_abmc(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    ///  Bandwidth counters can be configured to measure
    /// bandwidth consumed by a COS instead of an RMID if set
    pub fn has_select_cos(&self) -> bool {
        (self.ecx & 1) > 0
    }
}

/// Extended Feature Identification 2 (LEAF=0x8000_0021).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct ExtendedFeatureIdentification2 {
    eax: ExtendedFeatureIdentification2Eax,
    ebx: u32,
    _ecx: u32,
    _edx: u32,
}

impl ExtendedFeatureIdentification2 {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: ExtendedFeatureIdentification2Eax::from_bits_truncate(data.eax),
            ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// Processor ignores nested data breakpoints if set
    pub fn has_no_nested_data_bp(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::NO_NESTED_DATA_BP)
    }

    /// LFENCE is always dispatch serializing if set
    pub fn has_lfence_always_serializing(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::LFENCE_ALWAYS_SERIALIZING)
    }

    /// SMM paging configuration lock supported if set
    pub fn has_smm_pg_cfg_lock(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::SMM_PG_CFG_LOCK)
    }

    /// Null segment selector loads also clear the destination segment register
    /// base and limit supported if set
    pub fn has_null_select_clears_base(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::NULL_SELECT_CLEARS_BASE)
    }

    /// Upper Address Ignore is supported if set
    pub fn has_upper_address_ignore(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::UPPER_ADDRESS_IGNORE)
    }

    /// Automatic IBRS if set
    pub fn has_automatic_ibrs(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::AUTOMATIC_IBRS)
    }

    /// SMM_CTL MSR (C001_0116h) is not supported if set
    pub fn has_no_smm_ctl_msr(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::NO_SMM_CTL_MSR)
    }

    /// Prefetch control MSR supported if set
    pub fn has_prefetch_ctl_msr(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::PREFETCH_CTL_MSR)
    }

    /// CPUID disable for non-privileged software if set
    pub fn has_cpuid_user_dis(&self) -> bool {
        self.eax
            .contains(ExtendedFeatureIdentification2Eax::CPUID_USER_DIS)
    }

    /// The size of the Microcode patch in 16-byte multiples. If 0, the size of the
    /// patch is at most 5568 (15C0h) bytes.
    pub fn microcode_patch_size(&self) -> u16 {
        get_bits(self.ebx, 0, 11) as u16
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeatureIdentification2Eax: u32 {
        const NO_NESTED_DATA_BP = 1 << 0;
        const LFENCE_ALWAYS_SERIALIZING = 1 << 2;
        const SMM_PG_CFG_LOCK = 1 << 3;
        const NULL_SELECT_CLEARS_BASE = 1 << 6;
        const UPPER_ADDRESS_IGNORE = 1 << 7;
        const AUTOMATIC_IBRS = 1 << 8;
        const NO_SMM_CTL_MSR = 1 << 9;
        const PREFETCH_CTL_MSR = 1 << 13;
        const CPUID_USER_DIS = 1 << 17;
    }
}

/// Extended Performance Monitoring and Debug (LEAF=0x8000_0022).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct ExtendedPerformanceMonitoringDebug {
    eax: ExtendedPerformanceMonitoringDebugEax,
    ebx: u32,
    _ecx: u32,
    _edx: u32,
}

impl ExtendedPerformanceMonitoringDebug {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: ExtendedPerformanceMonitoringDebugEax::from_bits_truncate(data.eax),
            ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// Performance Monitoring Version 2 supported if set
    pub fn has_perf_mon_v2(&self) -> bool {
        self.eax
            .contains(ExtendedPerformanceMonitoringDebugEax::PERF_MON_V2)
    }

    /// Last Branch Record Stack supported if set
    pub fn has_lbr_stack(&self) -> bool {
        self.eax
            .contains(ExtendedPerformanceMonitoringDebugEax::LBR_STACK)
    }

    /// Freezing Core Performance Counters and LBR Stack on Core
    /// Performance Counter overflow supported if set
    pub fn has_lbr_and_pmc_freeze(&self) -> bool {
        self.eax
            .contains(ExtendedPerformanceMonitoringDebugEax::LBR_AND_PMC_FREEZE)
    }

    /// Number of Core Performance Counters
    pub fn num_perf_ctr_core(&self) -> u8 {
        get_bits(self.ebx, 0, 3) as u8
    }

    /// Number of Last Branch Record Stack entries
    pub fn num_lbr_stack_size(&self) -> u8 {
        get_bits(self.ebx, 4, 9) as u8
    }

    /// Number of Northbridge Performance Monitor Counters
    pub fn num_perf_ctr_nb(&self) -> u8 {
        get_bits(self.ebx, 10, 15) as u8
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedPerformanceMonitoringDebugEax: u32 {
        const PERF_MON_V2 = 1 << 0;
        const LBR_STACK = 1 << 1;
        const LBR_AND_PMC_FREEZE = 1 << 2;
    }
}

/// Multi-Key Encrypted Memory Capabilities (LEAF=0x8000_0023).
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(PartialEq, Eq, Debug)]
pub struct MultiKeyEncryptedMemoryCapabilities {
    eax: MultiKeyEncryptedMemoryCapabilitiesEax,
    ebx: u32,
    _ecx: u32,
    _edx: u32,
}

impl MultiKeyEncryptedMemoryCapabilities {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: MultiKeyEncryptedMemoryCapabilitiesEax::from_bits_truncate(data.eax),
            ebx: data.ebx,
            _ecx: data.ecx,
            _edx: data.edx,
        }
    }

    /// Secure Host Multi-Key Memory (MEM-HMK) Encryption Mode Supported if set
    pub fn has_mem_hmk(&self) -> bool {
        self.eax
            .contains(MultiKeyEncryptedMemoryCapabilitiesEax::MEM_HMK)
    }

    /// Number of simultaneously available host encryption key IDs in MEM-HMK
    /// encryption mode.
    pub fn max_mem_hmk_encr_key_id(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MultiKeyEncryptedMemoryCapabilitiesEax: u32 {
        const MEM_HMK = 1 << 0;
    }
}

/// Extended CPU Topology (LEAF=0x8000_0026).
///
/// Iterates over the extended cpu topology in order to retrieve more information for logical
/// processors, including asymmetric and heterogenous topology descriptions. Individual
/// logical processors may report different values in systems with asynchronous and
/// heterogeneous topologies
///
/// # Platforms
/// ✅ AMD ❌ Intel
#[derive(Clone)]
pub struct ExtendedCpuTopologyIter<R: CpuIdReader> {
    read: R,
    level: u32,
}

impl<R: CpuIdReader> ExtendedCpuTopologyIter<R> {
    pub fn new(read: R) -> Self {
        Self { read, level: 0 }
    }
}

/// Gives information about the current level in the cpu topology.
#[derive(PartialEq, Eq, Debug)]
pub struct ExtendedCpuTopologyLevel {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl ExtendedCpuTopologyLevel {
    pub(crate) fn new(data: CpuIdResult) -> Self {
        Self {
            eax: data.eax,
            ebx: data.ebx,
            ecx: data.ecx,
            edx: data.edx,
        }
    }

    /// Number of bits to shift Extended APIC ID right to get a unique topology ID
    /// of the current hierarchy level.
    pub fn mask_width(&self) -> u8 {
        get_bits(self.eax, 0, 4) as u8
    }

    /// Set to 1 if processor power efficiency ranking (PwrEfficiencyRanking) is
    /// available and varies between cores. Only valid for LevelType = 1h (Core).
    pub fn has_efficiency_ranking_available(&self) -> bool {
        self.eax & (1 << 29) > 0
    }

    /// Set to 1 if all components at the current hierarchy level do not consist of
    /// the cores that report the same core type (CoreType).
    pub fn has_heterogeneous_cores(&self) -> bool {
        self.eax & (1 << 30) > 0
    }

    /// Set to 1 if all components at the current hierarchy level do not report the
    /// same number of logical processors (NumLogProc).
    pub fn has_asymmetric_topology(&self) -> bool {
        self.eax & (1 << 31) > 0
    }

    /// Number of logical processors at the current hierarchy level
    pub fn num_logical_processors(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    /// Reports a static efficiency ranking between cores of a specific core type,
    /// where a lower value indicates comparatively lower power consumption
    /// and lower performance. Only valid for LevelType = 1h (Core)
    pub fn pwr_efficiency_ranking(&self) -> u8 {
        get_bits(self.ebx, 16, 23) as u8
    }

    /// Reports a value that may be used to further differentiate implementation
    /// specific features. Native mode ID is used in conjunction with the family,
    /// model, and stepping identifiers. Refer to the Processor Programming
    /// Reference Manual applicable to your product for a list of Native Mode
    /// IDs. Only valid for LevelType = 1h (Core)
    pub fn native_mode_id(&self) -> u8 {
        get_bits(self.ebx, 24, 27) as u8
    }

    /// Reports a value that may be used to distinguish between cores with
    /// different architectural and microarchitectural properties (for example,
    /// cores with different performance or power characteristics). Refer to the
    /// Processor Programming Reference Manual applicable to your product for
    /// a list of the available core types. Only valid for LevelType = 1h (Core)
    pub fn core_type(&self) -> u8 {
        get_bits(self.ebx, 28, 31) as u8
    }

    /// Input ECX
    pub fn input_ecx(&self) -> u8 {
        get_bits(self.ecx, 0, 7) as u8
    }

    /// Encoded hierarchy level type
    pub fn level_type(&self) -> HierarchyLevelType {
        HierarchyLevelType::from(get_bits(self.ecx, 8, 15) as u8)
    }

    /// Extended APIC ID of the logical processor
    pub fn extended_apic_id(&self) -> u32 {
        self.edx
    }
}

impl<R: CpuIdReader> Iterator for ExtendedCpuTopologyIter<R> {
    type Item = ExtendedCpuTopologyLevel;

    fn next(&mut self) -> Option<ExtendedCpuTopologyLevel> {
        let res = self.read.cpuid2(EAX_EXTENDED_CPU_TOPOLOGY, self.level);
        self.level += 1;

        let ect = ExtendedCpuTopologyLevel::new(res);
        if ect.level_type() == HierarchyLevelType::Reserved {
            None
        } else {
            Some(ect)
        }
    }
}

#[repr(u8)]
#[derive(PartialEq, Eq)]
pub enum HierarchyLevelType {
    Reserved = 0,
    Core = 1,
    Complex = 2,
    Die = 3,
    Socket = 4,
    Unknown(u8),
}

impl From<u8> for HierarchyLevelType {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Reserved,
            1 => Self::Core,
            2 => Self::Complex,
            3 => Self::Die,
            4 => Self::Socket,
            x => Self::Unknown(x),
        }
    }
}

impl Display for HierarchyLevelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            HierarchyLevelType::Reserved => write!(f, "Reserved (0)"),
            HierarchyLevelType::Core => write!(f, "Core (1)"),
            HierarchyLevelType::Complex => write!(f, "Complex (2)"),
            HierarchyLevelType::Die => write!(f, "DIE (3)"),
            HierarchyLevelType::Socket => write!(f, "Socket (4)"),
            HierarchyLevelType::Unknown(x) => write!(f, "Unknown ({x})"),
        }
    }
}

impl Debug for HierarchyLevelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reserved => write!(f, "Reserved"),
            Self::Core => write!(f, "Core"),
            Self::Complex => write!(f, "Complex"),
            Self::Die => write!(f, "Die"),
            Self::Socket => write!(f, "Socket"),
            Self::Unknown(arg0) => f.debug_tuple("Unknown").field(arg0).finish(),
        }
    }
}
