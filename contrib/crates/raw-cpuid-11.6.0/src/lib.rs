//! A library to parse the x86 CPUID instruction, written in rust with no
//! external dependencies. The implementation closely resembles the Intel CPUID
//! manual description. The library works with no_std.
//!
//! ## Example
//! ```rust
//! use raw_cpuid::CpuId;
//! let cpuid = CpuId::new();
//!
//! if let Some(vf) = cpuid.get_vendor_info() {
//!     assert!(vf.as_str() == "GenuineIntel" || vf.as_str() == "AuthenticAMD");
//! }
//!
//! let has_sse = cpuid.get_feature_info().map_or(false, |finfo| finfo.has_sse());
//! if has_sse {
//!     println!("CPU supports SSE!");
//! }
//!
//! if let Some(cparams) = cpuid.get_cache_parameters() {
//!     for cache in cparams {
//!         let size = cache.associativity() * cache.physical_line_partitions() * cache.coherency_line_size() * cache.sets();
//!         println!("L{}-Cache size is {}", cache.level(), size);
//!     }
//! } else {
//!     println!("No cache parameter information available")
//! }
//! ```
//!
//! # Platform support
//!
//! CPU vendors may choose to not support certain functions/leafs in cpuid or
//! only support them partially. We highlight this with the following emojis
//! throughout the documentation:
//!
//! - ✅: This struct/function is fully supported by the vendor.
//! - 🟡: This struct is partially supported by the vendor, refer to individual
//!   functions for more information.
//! - ❌: This struct/function is not supported by the vendor. When queried on
//!   this platform, we will return None/false/0 (or some other sane default).
//! - ❓: This struct/function is not supported by the vendor according to the
//!   manual, but the in practice it still may return valid information.
//!
//! Note that the presence of a ✅ does not guarantee that a specific feature
//! will exist for your CPU -- just that it is potentially supported by the
//! vendor on some of its chips. You will still have to query it at runtime.

#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "raw_cpuid"]
#![crate_type = "lib"]

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(feature = "display")]
pub mod display;
mod extended;
#[cfg(test)]
mod tests;

use bitflags::bitflags;
use core::fmt::{self, Debug, Formatter};
use core::mem::size_of;
use core::slice;
use core::str;

#[cfg(feature = "serialize")]
use serde_derive::{Deserialize, Serialize};

pub use extended::*;

/// Uses Rust's `cpuid` function from the `arch` module.
#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
pub mod native_cpuid {
    use crate::CpuIdResult;

    #[cfg(all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"))]
    use core::arch::x86 as arch;
    #[cfg(all(target_arch = "x86_64", not(target_env = "sgx")))]
    use core::arch::x86_64 as arch;

    pub fn cpuid_count(a: u32, c: u32) -> CpuIdResult {
        // Safety: CPUID is supported on all x86_64 CPUs and all x86 CPUs with
        // SSE, but not by SGX.
        let result = unsafe { self::arch::__cpuid_count(a, c) };

        CpuIdResult {
            eax: result.eax,
            ebx: result.ebx,
            ecx: result.ecx,
            edx: result.edx,
        }
    }
    /// The native reader uses the cpuid instruction to read the cpuid data from the
    /// CPU we're currently running on directly.
    #[derive(Clone, Copy)]
    pub struct CpuIdReaderNative;

    impl super::CpuIdReader for CpuIdReaderNative {
        fn cpuid2(&self, eax: u32, ecx: u32) -> CpuIdResult {
            cpuid_count(eax, ecx)
        }
    }
}

#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
pub use native_cpuid::CpuIdReaderNative;

/// Macro which queries cpuid directly.
///
/// First parameter is cpuid leaf (EAX register value),
/// second optional parameter is the subleaf (ECX register value).
#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
#[macro_export]
macro_rules! cpuid {
    ($eax:expr) => {
        $crate::native_cpuid::cpuid_count($eax as u32, 0)
    };

    ($eax:expr, $ecx:expr) => {
        $crate::native_cpuid::cpuid_count($eax as u32, $ecx as u32)
    };
}

fn get_bits(r: u32, from: u32, to: u32) -> u32 {
    assert!(from <= 31);
    assert!(to <= 31);
    assert!(from <= to);

    let mask = match to {
        31 => 0xffffffff,
        _ => (1 << (to + 1)) - 1,
    };

    (r & mask) >> from
}

macro_rules! check_flag {
    ($doc:meta, $fun:ident, $flags:ident, $flag:expr) => {
        #[$doc]
        pub fn $fun(&self) -> bool {
            self.$flags.contains($flag)
        }
    };
}

macro_rules! is_bit_set {
    ($field:expr, $bit:expr) => {
        $field & (1 << $bit) > 0
    };
}

macro_rules! check_bit_fn {
    ($doc:meta, $fun:ident, $field:ident, $bit:expr) => {
        #[$doc]
        pub fn $fun(&self) -> bool {
            is_bit_set!(self.$field, $bit)
        }
    };
}

/// Implements function to read/write cpuid.
/// This allows to conveniently swap out the underlying cpuid implementation
/// with one that returns data that is deterministic (for unit-testing).
pub trait CpuIdReader: Clone {
    fn cpuid1(&self, eax: u32) -> CpuIdResult {
        self.cpuid2(eax, 0)
    }
    fn cpuid2(&self, eax: u32, ecx: u32) -> CpuIdResult;
}

impl<F> CpuIdReader for F
where
    F: Fn(u32, u32) -> CpuIdResult + Clone,
{
    fn cpuid2(&self, eax: u32, ecx: u32) -> CpuIdResult {
        self(eax, ecx)
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum Vendor {
    Intel,
    Amd,
    Unknown(u32, u32, u32),
}

impl Vendor {
    fn from_vendor_leaf(res: CpuIdResult) -> Self {
        let vi = VendorInfo {
            ebx: res.ebx,
            ecx: res.ecx,
            edx: res.edx,
        };

        match vi.as_str() {
            "GenuineIntel" => Vendor::Intel,
            "AuthenticAMD" | "HygonGenuine" => Vendor::Amd,
            _ => Vendor::Unknown(res.ebx, res.ecx, res.edx),
        }
    }
}

/// The main type used to query information about the CPU we're running on.
///
/// Other structs can be accessed by going through this type.
#[derive(Clone, Copy)]
pub struct CpuId<R: CpuIdReader> {
    /// A generic reader to abstract the cpuid interface.
    read: R,
    /// CPU vendor to differentiate cases where logic needs to differ in code .
    vendor: Vendor,
    /// How many basic leafs are supported (EAX < EAX_HYPERVISOR_INFO)
    supported_leafs: u32,
    /// How many extended leafs are supported (e.g., leafs with EAX > EAX_EXTENDED_FUNCTION_INFO)
    supported_extended_leafs: u32,
}

#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
impl Default for CpuId<CpuIdReaderNative> {
    /// Create a new `CpuId` instance.
    fn default() -> Self {
        CpuId::with_cpuid_fn(CpuIdReaderNative)
    }
}

#[cfg(any(
    all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
    all(target_arch = "x86_64", not(target_env = "sgx"))
))]
impl CpuId<CpuIdReaderNative> {
    /// Create a new `CpuId` instance.
    pub fn new() -> Self {
        CpuId::default()
    }
}

/// Low-level data-structure to store result of cpuid instruction.
#[derive(Copy, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct CpuIdResult {
    /// Return value EAX register
    pub eax: u32,
    /// Return value EBX register
    pub ebx: u32,
    /// Return value ECX register
    pub ecx: u32,
    /// Return value EDX register
    pub edx: u32,
}

impl CpuIdResult {
    pub fn all_zero(&self) -> bool {
        self.eax == 0 && self.ebx == 0 && self.ecx == 0 && self.edx == 0
    }
}

impl Debug for CpuIdResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuIdResult")
            .field("eax", &(self.eax as *const u32))
            .field("ebx", &(self.ebx as *const u32))
            .field("ecx", &(self.ecx as *const u32))
            .field("edx", &(self.edx as *const u32))
            .finish()
    }
}

//
// Normal leafs:
//
const EAX_VENDOR_INFO: u32 = 0x0;
const EAX_FEATURE_INFO: u32 = 0x1;
const EAX_CACHE_INFO: u32 = 0x2;
const EAX_PROCESSOR_SERIAL: u32 = 0x3;
const EAX_CACHE_PARAMETERS: u32 = 0x4;
const EAX_MONITOR_MWAIT_INFO: u32 = 0x5;
const EAX_THERMAL_POWER_INFO: u32 = 0x6;
const EAX_STRUCTURED_EXTENDED_FEATURE_INFO: u32 = 0x7;
const EAX_DIRECT_CACHE_ACCESS_INFO: u32 = 0x9;
const EAX_PERFORMANCE_MONITOR_INFO: u32 = 0xA;
const EAX_EXTENDED_TOPOLOGY_INFO: u32 = 0xB;
const EAX_EXTENDED_STATE_INFO: u32 = 0xD;
const EAX_RDT_MONITORING: u32 = 0xF;
const EAX_RDT_ALLOCATION: u32 = 0x10;
const EAX_SGX: u32 = 0x12;
const EAX_TRACE_INFO: u32 = 0x14;
const EAX_TIME_STAMP_COUNTER_INFO: u32 = 0x15;
const EAX_FREQUENCY_INFO: u32 = 0x16;
const EAX_SOC_VENDOR_INFO: u32 = 0x17;
const EAX_DETERMINISTIC_ADDRESS_TRANSLATION_INFO: u32 = 0x18;
const EAX_EXTENDED_TOPOLOGY_INFO_V2: u32 = 0x1F;

/// Hypervisor leaf
const EAX_HYPERVISOR_INFO: u32 = 0x4000_0000;

//
// Extended leafs:
//
const EAX_EXTENDED_FUNCTION_INFO: u32 = 0x8000_0000;
const EAX_EXTENDED_PROCESSOR_AND_FEATURE_IDENTIFIERS: u32 = 0x8000_0001;
const EAX_EXTENDED_BRAND_STRING: u32 = 0x8000_0002;
const EAX_L1_CACHE_INFO: u32 = 0x8000_0005;
const EAX_L2_L3_CACHE_INFO: u32 = 0x8000_0006;
const EAX_ADVANCED_POWER_MGMT_INFO: u32 = 0x8000_0007;
const EAX_PROCESSOR_CAPACITY_INFO: u32 = 0x8000_0008;
const EAX_TLB_1GB_PAGE_INFO: u32 = 0x8000_0019;
const EAX_PERFORMANCE_OPTIMIZATION_INFO: u32 = 0x8000_001A;
const EAX_INSTRUCTION_BASED_SAMPLING_CAPABILITIES: u32 = 0x8000_001B;
const EAX_CACHE_PARAMETERS_AMD: u32 = 0x8000_001D;
const EAX_PROCESSOR_TOPOLOGY_INFO: u32 = 0x8000_001E;
const EAX_MEMORY_ENCRYPTION_INFO: u32 = 0x8000_001F;
const EAX_SVM_FEATURES: u32 = 0x8000_000A;
const EAX_PQOS_EXTENDED_FEATURES: u32 = 0x8000_0020;
const EAX_EXTENDED_FEATURE_IDENTIFICATION_2: u32 = 0x8000_0021;
const EAX_EXTENDED_PERFORMANCE_MONITORING_AND_DEBUG: u32 = 0x8000_0022;
const EAX_MULTI_KEY_ENCRYPTED_MEMORY_CAPABILITIES: u32 = 0x8000_0023;
const EAX_EXTENDED_CPU_TOPOLOGY: u32 = 0x8000_0026;

impl<R: CpuIdReader> CpuId<R> {
    /// Return new CpuId struct with custom reader function.
    ///
    /// This is useful for example when testing code or if we want to interpose
    /// on the CPUID calls this library makes.
    pub fn with_cpuid_reader(cpuid_fn: R) -> Self {
        let vendor_leaf = cpuid_fn.cpuid1(EAX_VENDOR_INFO);
        let extended_leaf = cpuid_fn.cpuid1(EAX_EXTENDED_FUNCTION_INFO);
        CpuId {
            supported_leafs: vendor_leaf.eax,
            supported_extended_leafs: extended_leaf.eax,
            vendor: Vendor::from_vendor_leaf(vendor_leaf),
            read: cpuid_fn,
        }
    }

    /// See [`CpuId::with_cpuid_reader`].
    ///
    /// # Note
    /// This function will likely be deprecated in the future. Use the identical
    /// `with_cpuid_reader` function instead.
    pub fn with_cpuid_fn(cpuid_fn: R) -> Self {
        CpuId::with_cpuid_reader(cpuid_fn)
    }

    /// Check if a non extended leaf  (`val`) is supported.
    fn leaf_is_supported(&self, val: u32) -> bool {
        // Exclude reserved functions/leafs on AMD
        if self.vendor == Vendor::Amd && ((0x2..=0x4).contains(&val) || (0x8..=0xa).contains(&val))
        {
            return false;
        }

        if val < EAX_EXTENDED_FUNCTION_INFO {
            val <= self.supported_leafs
        } else {
            val <= self.supported_extended_leafs
        }
    }

    /// Return information about the vendor (LEAF=0x00).
    ///
    /// This leaf will contain a ASCII readable string such as "GenuineIntel"
    /// for Intel CPUs or "AuthenticAMD" for AMD CPUs.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn get_vendor_info(&self) -> Option<VendorInfo> {
        if self.leaf_is_supported(EAX_VENDOR_INFO) {
            let res = self.read.cpuid1(EAX_VENDOR_INFO);
            Some(VendorInfo {
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Query a set of features that are available on this CPU (LEAF=0x01).
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn get_feature_info(&self) -> Option<FeatureInfo> {
        if self.leaf_is_supported(EAX_FEATURE_INFO) {
            let res = self.read.cpuid1(EAX_FEATURE_INFO);
            Some(FeatureInfo {
                vendor: self.vendor,
                eax: res.eax,
                ebx: res.ebx,
                edx_ecx: FeatureInfoFlags::from_bits_truncate(
                    ((res.edx as u64) << 32) | (res.ecx as u64),
                ),
            })
        } else {
            None
        }
    }

    /// Query basic information about caches (LEAF=0x02).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_cache_info(&self) -> Option<CacheInfoIter> {
        if self.leaf_is_supported(EAX_CACHE_INFO) {
            let res = self.read.cpuid1(EAX_CACHE_INFO);
            Some(CacheInfoIter {
                current: 1,
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Retrieve serial number of processor (LEAF=0x03).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_processor_serial(&self) -> Option<ProcessorSerial> {
        if self.leaf_is_supported(EAX_PROCESSOR_SERIAL) {
            // upper 64-96 bits are in res1.eax:
            let res1 = self.read.cpuid1(EAX_FEATURE_INFO);
            let res = self.read.cpuid1(EAX_PROCESSOR_SERIAL);
            Some(ProcessorSerial {
                ecx: res.ecx,
                edx: res.edx,
                eax: res1.eax,
            })
        } else {
            None
        }
    }

    /// Retrieve more elaborate information about caches (LEAF=0x04 or 0x8000_001D).
    ///
    /// As opposed to [get_cache_info](CpuId::get_cache_info), this will tell us
    /// about associativity, set size, line size of each level in the cache
    /// hierarchy.
    ///
    /// # Platforms
    /// 🟡 AMD ✅ Intel
    pub fn get_cache_parameters(&self) -> Option<CacheParametersIter<R>> {
        if self.leaf_is_supported(EAX_CACHE_PARAMETERS)
            || (self.vendor == Vendor::Amd && self.leaf_is_supported(EAX_CACHE_PARAMETERS_AMD))
        {
            Some(CacheParametersIter {
                read: self.read.clone(),
                leaf: if self.vendor == Vendor::Amd {
                    EAX_CACHE_PARAMETERS_AMD
                } else {
                    EAX_CACHE_PARAMETERS
                },
                current: 0,
            })
        } else {
            None
        }
    }

    /// Information about how monitor/mwait works on this CPU (LEAF=0x05).
    ///
    /// # Platforms
    /// 🟡 AMD ✅ Intel
    pub fn get_monitor_mwait_info(&self) -> Option<MonitorMwaitInfo> {
        if self.leaf_is_supported(EAX_MONITOR_MWAIT_INFO) {
            let res = self.read.cpuid1(EAX_MONITOR_MWAIT_INFO);
            Some(MonitorMwaitInfo {
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Query information about thermal and power management features of the CPU (LEAF=0x06).
    ///
    /// # Platforms
    /// 🟡 AMD ✅ Intel
    pub fn get_thermal_power_info(&self) -> Option<ThermalPowerInfo> {
        if self.leaf_is_supported(EAX_THERMAL_POWER_INFO) {
            let res = self.read.cpuid1(EAX_THERMAL_POWER_INFO);
            Some(ThermalPowerInfo {
                eax: ThermalPowerFeaturesEax::from_bits_truncate(res.eax),
                ebx: res.ebx,
                ecx: ThermalPowerFeaturesEcx::from_bits_truncate(res.ecx),
                _edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Find out about more features supported by this CPU (LEAF=0x07).
    ///
    /// # Platforms
    /// 🟡 AMD ✅ Intel
    pub fn get_extended_feature_info(&self) -> Option<ExtendedFeatures> {
        if self.leaf_is_supported(EAX_STRUCTURED_EXTENDED_FEATURE_INFO) {
            let res = self.read.cpuid1(EAX_STRUCTURED_EXTENDED_FEATURE_INFO);
            let res1 = self.read.cpuid2(EAX_STRUCTURED_EXTENDED_FEATURE_INFO, 1);
            Some(ExtendedFeatures {
                _eax: res.eax,
                ebx: ExtendedFeaturesEbx::from_bits_truncate(res.ebx),
                ecx: ExtendedFeaturesEcx::from_bits_truncate(res.ecx),
                edx: ExtendedFeaturesEdx::from_bits_truncate(res.edx),
                eax1: ExtendedFeaturesEax1::from_bits_truncate(res1.eax),
                _ebx1: res1.ebx,
                _ecx1: res1.ecx,
                edx1: ExtendedFeaturesEdx1::from_bits_truncate(res1.edx),
            })
        } else {
            None
        }
    }

    /// Direct cache access info (LEAF=0x09).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_direct_cache_access_info(&self) -> Option<DirectCacheAccessInfo> {
        if self.leaf_is_supported(EAX_DIRECT_CACHE_ACCESS_INFO) {
            let res = self.read.cpuid1(EAX_DIRECT_CACHE_ACCESS_INFO);
            Some(DirectCacheAccessInfo { eax: res.eax })
        } else {
            None
        }
    }

    /// Info about performance monitoring (LEAF=0x0A).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_performance_monitoring_info(&self) -> Option<PerformanceMonitoringInfo> {
        if self.leaf_is_supported(EAX_PERFORMANCE_MONITOR_INFO) {
            let res = self.read.cpuid1(EAX_PERFORMANCE_MONITOR_INFO);
            Some(PerformanceMonitoringInfo {
                eax: res.eax,
                ebx: PerformanceMonitoringFeaturesEbx::from_bits_truncate(res.ebx),
                _ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Information about topology (LEAF=0x0B).
    ///
    /// Intel SDM suggests software should check support for leaf 0x1F
    /// ([`CpuId::get_extended_topology_info_v2`]), and if supported, enumerate
    /// that leaf instead.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn get_extended_topology_info(&self) -> Option<ExtendedTopologyIter<R>> {
        if self.leaf_is_supported(EAX_EXTENDED_TOPOLOGY_INFO) {
            Some(ExtendedTopologyIter {
                read: self.read.clone(),
                level: 0,
                is_v2: false,
            })
        } else {
            None
        }
    }

    /// Extended information about topology (LEAF=0x1F).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_extended_topology_info_v2(&self) -> Option<ExtendedTopologyIter<R>> {
        if self.leaf_is_supported(EAX_EXTENDED_TOPOLOGY_INFO_V2) {
            Some(ExtendedTopologyIter {
                read: self.read.clone(),
                level: 0,
                is_v2: true,
            })
        } else {
            None
        }
    }

    /// Information for saving/restoring extended register state (LEAF=0x0D).
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn get_extended_state_info(&self) -> Option<ExtendedStateInfo<R>> {
        if self.leaf_is_supported(EAX_EXTENDED_STATE_INFO) {
            let res = self.read.cpuid2(EAX_EXTENDED_STATE_INFO, 0);
            let res1 = self.read.cpuid2(EAX_EXTENDED_STATE_INFO, 1);
            Some(ExtendedStateInfo {
                read: self.read.clone(),
                eax: ExtendedStateInfoXCR0Flags::from_bits_truncate(res.eax),
                ebx: res.ebx,
                ecx: res.ecx,
                _edx: res.edx,
                eax1: res1.eax,
                ebx1: res1.ebx,
                ecx1: ExtendedStateInfoXSSFlags::from_bits_truncate(res1.ecx),
                _edx1: res1.edx,
            })
        } else {
            None
        }
    }

    /// Quality of service monitoring information (LEAF=0x0F).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_rdt_monitoring_info(&self) -> Option<RdtMonitoringInfo<R>> {
        let res = self.read.cpuid1(EAX_RDT_MONITORING);

        if self.leaf_is_supported(EAX_RDT_MONITORING) {
            Some(RdtMonitoringInfo {
                read: self.read.clone(),
                ebx: res.ebx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Quality of service enforcement information (LEAF=0x10).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_rdt_allocation_info(&self) -> Option<RdtAllocationInfo<R>> {
        let res = self.read.cpuid1(EAX_RDT_ALLOCATION);

        if self.leaf_is_supported(EAX_RDT_ALLOCATION) {
            Some(RdtAllocationInfo {
                read: self.read.clone(),
                ebx: res.ebx,
            })
        } else {
            None
        }
    }

    /// Information about secure enclave support (LEAF=0x12).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_sgx_info(&self) -> Option<SgxInfo<R>> {
        // Leaf 12H sub-leaf 0 (ECX = 0) is supported if CPUID.(EAX=07H, ECX=0H):EBX[SGX] = 1.
        self.get_extended_feature_info().and_then(|info| {
            if self.leaf_is_supported(EAX_SGX) && info.has_sgx() {
                let res = self.read.cpuid2(EAX_SGX, 0);
                let res1 = self.read.cpuid2(EAX_SGX, 1);
                Some(SgxInfo {
                    read: self.read.clone(),
                    eax: res.eax,
                    ebx: res.ebx,
                    _ecx: res.ecx,
                    edx: res.edx,
                    eax1: res1.eax,
                    ebx1: res1.ebx,
                    ecx1: res1.ecx,
                    edx1: res1.edx,
                })
            } else {
                None
            }
        })
    }

    /// Intel Processor Trace Enumeration Information (LEAF=0x14).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_processor_trace_info(&self) -> Option<ProcessorTraceInfo> {
        if self.leaf_is_supported(EAX_TRACE_INFO) {
            let res = self.read.cpuid2(EAX_TRACE_INFO, 0);
            let res1 = if res.eax >= 1 {
                Some(self.read.cpuid2(EAX_TRACE_INFO, 1))
            } else {
                None
            };

            Some(ProcessorTraceInfo {
                _eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                _edx: res.edx,
                leaf1: res1,
            })
        } else {
            None
        }
    }

    /// Time Stamp Counter/Core Crystal Clock Information (LEAF=0x15).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_tsc_info(&self) -> Option<TscInfo> {
        if self.leaf_is_supported(EAX_TIME_STAMP_COUNTER_INFO) {
            let res = self.read.cpuid2(EAX_TIME_STAMP_COUNTER_INFO, 0);
            Some(TscInfo {
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
            })
        } else {
            None
        }
    }

    /// Processor Frequency Information (LEAF=0x16).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_processor_frequency_info(&self) -> Option<ProcessorFrequencyInfo> {
        if self.leaf_is_supported(EAX_FREQUENCY_INFO) {
            let res = self.read.cpuid1(EAX_FREQUENCY_INFO);
            Some(ProcessorFrequencyInfo {
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
            })
        } else {
            None
        }
    }

    /// Contains SoC vendor specific information (LEAF=0x17).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_soc_vendor_info(&self) -> Option<SoCVendorInfo<R>> {
        if self.leaf_is_supported(EAX_SOC_VENDOR_INFO) {
            let res = self.read.cpuid1(EAX_SOC_VENDOR_INFO);
            Some(SoCVendorInfo {
                read: self.read.clone(),
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Query deterministic address translation feature (LEAF=0x18).
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn get_deterministic_address_translation_info(&self) -> Option<DatIter<R>> {
        if self.leaf_is_supported(EAX_DETERMINISTIC_ADDRESS_TRANSLATION_INFO) {
            let res = self
                .read
                .cpuid2(EAX_DETERMINISTIC_ADDRESS_TRANSLATION_INFO, 0);
            Some(DatIter {
                read: self.read.clone(),
                current: 0,
                count: res.eax,
            })
        } else {
            None
        }
    }

    /// Returns information provided by the hypervisor, if running
    /// in a virtual environment (LEAF=0x4000_00xx).
    ///
    /// # Platform
    /// Needs to be a virtual CPU to be supported.
    pub fn get_hypervisor_info(&self) -> Option<HypervisorInfo<R>> {
        // We only fetch HypervisorInfo, if the Hypervisor-Flag is set.
        // See https://github.com/gz/rust-cpuid/issues/52
        self.get_feature_info()
            .filter(|fi| fi.has_hypervisor())
            .and_then(|_| {
                let res = self.read.cpuid1(EAX_HYPERVISOR_INFO);
                if res.eax > 0 {
                    Some(HypervisorInfo {
                        read: self.read.clone(),
                        res,
                    })
                } else {
                    None
                }
            })
    }

    /// Extended Processor and Processor Feature Identifiers (LEAF=0x8000_0001).
    ///
    /// # Platforms
    /// ✅ AMD 🟡 Intel
    pub fn get_extended_processor_and_feature_identifiers(
        &self,
    ) -> Option<ExtendedProcessorFeatureIdentifiers> {
        if self.leaf_is_supported(EAX_EXTENDED_PROCESSOR_AND_FEATURE_IDENTIFIERS) {
            Some(ExtendedProcessorFeatureIdentifiers::new(
                self.vendor,
                self.read
                    .cpuid1(EAX_EXTENDED_PROCESSOR_AND_FEATURE_IDENTIFIERS),
            ))
        } else {
            None
        }
    }

    /// Retrieve processor brand string (LEAF=0x8000_000{2..4}).
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn get_processor_brand_string(&self) -> Option<ProcessorBrandString> {
        if self.leaf_is_supported(EAX_EXTENDED_BRAND_STRING)
            && self.leaf_is_supported(EAX_EXTENDED_BRAND_STRING + 1)
            && self.leaf_is_supported(EAX_EXTENDED_BRAND_STRING + 2)
        {
            Some(ProcessorBrandString::new([
                self.read.cpuid1(EAX_EXTENDED_BRAND_STRING),
                self.read.cpuid1(EAX_EXTENDED_BRAND_STRING + 1),
                self.read.cpuid1(EAX_EXTENDED_BRAND_STRING + 2),
            ]))
        } else {
            None
        }
    }

    /// L1 Instruction Cache Information (LEAF=0x8000_0005)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_l1_cache_and_tlb_info(&self) -> Option<L1CacheTlbInfo> {
        if self.vendor == Vendor::Amd && self.leaf_is_supported(EAX_L1_CACHE_INFO) {
            Some(L1CacheTlbInfo::new(self.read.cpuid1(EAX_L1_CACHE_INFO)))
        } else {
            None
        }
    }

    /// L2/L3 Cache and TLB Information (LEAF=0x8000_0006).
    ///
    /// # Platforms
    /// ✅ AMD 🟡 Intel
    pub fn get_l2_l3_cache_and_tlb_info(&self) -> Option<L2And3CacheTlbInfo> {
        if self.leaf_is_supported(EAX_L2_L3_CACHE_INFO) {
            Some(L2And3CacheTlbInfo::new(
                self.read.cpuid1(EAX_L2_L3_CACHE_INFO),
            ))
        } else {
            None
        }
    }

    /// Advanced Power Management Information (LEAF=0x8000_0007).
    ///
    /// # Platforms
    /// ✅ AMD 🟡 Intel
    pub fn get_advanced_power_mgmt_info(&self) -> Option<ApmInfo> {
        if self.leaf_is_supported(EAX_ADVANCED_POWER_MGMT_INFO) {
            Some(ApmInfo::new(self.read.cpuid1(EAX_ADVANCED_POWER_MGMT_INFO)))
        } else {
            None
        }
    }

    /// Processor Capacity Parameters and Extended Feature Identification (LEAF=0x8000_0008).
    ///
    /// # Platforms
    /// ✅ AMD 🟡 Intel
    pub fn get_processor_capacity_feature_info(&self) -> Option<ProcessorCapacityAndFeatureInfo> {
        if self.leaf_is_supported(EAX_PROCESSOR_CAPACITY_INFO) {
            Some(ProcessorCapacityAndFeatureInfo::new(
                self.read.cpuid1(EAX_PROCESSOR_CAPACITY_INFO),
            ))
        } else {
            None
        }
    }

    /// This function provides information about the SVM features that the processor
    /// supports. (LEAF=0x8000_000A)
    ///
    /// If SVM is not supported if [ExtendedProcessorFeatureIdentifiers::has_svm] is
    /// false, this function is reserved then.
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel
    pub fn get_svm_info(&self) -> Option<SvmFeatures> {
        let has_svm = self
            .get_extended_processor_and_feature_identifiers()
            .map_or(false, |f| f.has_svm());
        if has_svm && self.leaf_is_supported(EAX_SVM_FEATURES) {
            Some(SvmFeatures::new(self.read.cpuid1(EAX_SVM_FEATURES)))
        } else {
            None
        }
    }

    /// TLB 1-GiB Pages Information (LEAF=0x8000_0019)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel
    pub fn get_tlb_1gb_page_info(&self) -> Option<Tlb1gbPageInfo> {
        if self.leaf_is_supported(EAX_TLB_1GB_PAGE_INFO) {
            Some(Tlb1gbPageInfo::new(self.read.cpuid1(EAX_TLB_1GB_PAGE_INFO)))
        } else {
            None
        }
    }

    /// Informations about performance optimization (LEAF=0x8000_001A)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_performance_optimization_info(&self) -> Option<PerformanceOptimizationInfo> {
        if self.leaf_is_supported(EAX_PERFORMANCE_OPTIMIZATION_INFO) {
            Some(PerformanceOptimizationInfo::new(
                self.read.cpuid1(EAX_PERFORMANCE_OPTIMIZATION_INFO),
            ))
        } else {
            None
        }
    }

    /// Instruction-Based Sampling Capabilities (LEAF=0x8000_001B)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_instruction_based_sampling_capabilities(
        &self,
    ) -> Option<InstructionBasedSamplingCapabilities> {
        if self.leaf_is_supported(EAX_INSTRUCTION_BASED_SAMPLING_CAPABILITIES) {
            Some(InstructionBasedSamplingCapabilities::new(
                self.read
                    .cpuid1(EAX_INSTRUCTION_BASED_SAMPLING_CAPABILITIES),
            ))
        } else {
            None
        }
    }

    /// Informations about processor topology (LEAF=0x8000_001E)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_processor_topology_info(&self) -> Option<ProcessorTopologyInfo> {
        if self.leaf_is_supported(EAX_PROCESSOR_TOPOLOGY_INFO) {
            Some(ProcessorTopologyInfo::new(
                self.read.cpuid1(EAX_PROCESSOR_TOPOLOGY_INFO),
            ))
        } else {
            None
        }
    }

    /// Informations about memory encryption support (LEAF=0x8000_001F)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_memory_encryption_info(&self) -> Option<MemoryEncryptionInfo> {
        if self.leaf_is_supported(EAX_MEMORY_ENCRYPTION_INFO) {
            Some(MemoryEncryptionInfo::new(
                self.read.cpuid1(EAX_MEMORY_ENCRYPTION_INFO),
            ))
        } else {
            None
        }
    }

    /// Informations about Platform Quality of Service (LEAF=0x8000_0020)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_pqos_extended_feature_info(&self) -> Option<PqosExtendedFeatureInfo<R>> {
        if self.leaf_is_supported(EAX_PQOS_EXTENDED_FEATURES) {
            Some(PqosExtendedFeatureInfo::new(self.read.clone()))
        } else {
            None
        }
    }

    /// Extended Feature Identification 2 (LEAF=0x8000_0021)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_extended_feature_identification_2(&self) -> Option<ExtendedFeatureIdentification2> {
        if self.leaf_is_supported(EAX_EXTENDED_FEATURE_IDENTIFICATION_2) {
            Some(ExtendedFeatureIdentification2::new(
                self.read.cpuid1(EAX_EXTENDED_FEATURE_IDENTIFICATION_2),
            ))
        } else {
            None
        }
    }

    /// Extended Performance Monitoring and Debug (LEAF=0x8000_0022)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_extended_performance_monitoring_and_debug(
        &self,
    ) -> Option<ExtendedPerformanceMonitoringDebug> {
        if self.leaf_is_supported(EAX_EXTENDED_PERFORMANCE_MONITORING_AND_DEBUG) {
            Some(ExtendedPerformanceMonitoringDebug::new(
                self.read
                    .cpuid1(EAX_EXTENDED_PERFORMANCE_MONITORING_AND_DEBUG),
            ))
        } else {
            None
        }
    }

    /// Multi-Key Encrypted Memory Capabilities (LEAF=0x8000_0023)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_multi_key_encrypted_memory_capabilities(
        &self,
    ) -> Option<MultiKeyEncryptedMemoryCapabilities> {
        if self.leaf_is_supported(EAX_MULTI_KEY_ENCRYPTED_MEMORY_CAPABILITIES) {
            Some(MultiKeyEncryptedMemoryCapabilities::new(
                self.read
                    .cpuid1(EAX_MULTI_KEY_ENCRYPTED_MEMORY_CAPABILITIES),
            ))
        } else {
            None
        }
    }

    /// Extended CPU Topology (LEAF=0x8000_0026)
    ///
    /// # Platforms
    /// ✅ AMD ❌ Intel (reserved)
    pub fn get_extended_cpu_topology(&self) -> Option<ExtendedCpuTopologyIter<R>> {
        if self.leaf_is_supported(EAX_EXTENDED_CPU_TOPOLOGY) {
            Some(ExtendedCpuTopologyIter::new(self.read.clone()))
        } else {
            None
        }
    }
}

impl<R: CpuIdReader> Debug for CpuId<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuId")
            .field("vendor", &self.vendor)
            // .field("supported_leafs", &(self.supported_leafs as *const u32))
            // .field("supported_extended_leafs", &(self.supported_extended_leafs as *const u32))
            .field("vendor_info", &self.get_vendor_info())
            .field("feature_info", &self.get_feature_info())
            .field("cache_info", &self.get_cache_info())
            .field("processor_serial", &self.get_processor_serial())
            .field("cache_parameters", &self.get_cache_parameters())
            .field("monitor_mwait_info", &self.get_monitor_mwait_info())
            .field("thermal_power_info", &self.get_thermal_power_info())
            .field("extended_feature_info", &self.get_extended_feature_info())
            .field(
                "direct_cache_access_info",
                &self.get_direct_cache_access_info(),
            )
            .field(
                "performance_monitoring_info",
                &self.get_performance_monitoring_info(),
            )
            .field("extended_topology_info", &self.get_extended_topology_info())
            .field("extended_state_info", &self.get_extended_state_info())
            .field("rdt_monitoring_info", &self.get_rdt_monitoring_info())
            .field("rdt_allocation_info", &self.get_rdt_allocation_info())
            .field("sgx_info", &self.get_sgx_info())
            .field("processor_trace_info", &self.get_processor_trace_info())
            .field("tsc_info", &self.get_tsc_info())
            .field(
                "processor_frequency_info",
                &self.get_processor_frequency_info(),
            )
            .field(
                "deterministic_address_translation_info",
                &self.get_deterministic_address_translation_info(),
            )
            .field("soc_vendor_info", &self.get_soc_vendor_info())
            .field("hypervisor_info", &self.get_hypervisor_info())
            .field(
                "extended_processor_and_feature_identifiers",
                &self.get_extended_processor_and_feature_identifiers(),
            )
            .field("processor_brand_string", &self.get_processor_brand_string())
            .field("l1_cache_and_tlb_info", &self.get_l1_cache_and_tlb_info())
            .field(
                "l2_l3_cache_and_tlb_info",
                &self.get_l2_l3_cache_and_tlb_info(),
            )
            .field(
                "advanced_power_mgmt_info",
                &self.get_advanced_power_mgmt_info(),
            )
            .field(
                "processor_capacity_feature_info",
                &self.get_processor_capacity_feature_info(),
            )
            .field("svm_info", &self.get_svm_info())
            .field("tlb_1gb_page_info", &self.get_tlb_1gb_page_info())
            .field(
                "performance_optimization_info",
                &self.get_performance_optimization_info(),
            )
            .field(
                "processor_topology_info",
                &self.get_processor_topology_info(),
            )
            .field("memory_encryption_info", &self.get_memory_encryption_info())
            .finish()
    }
}

/// Vendor Info String (LEAF=0x0)
///
/// A string that can be for example "AuthenticAMD" or "GenuineIntel".
///
/// # Technical Background
///
/// The vendor info is a 12-byte (96 bit) long string stored in `ebx`, `edx` and
/// `ecx` by the corresponding `cpuid` instruction.
///
/// # Platforms
/// ✅ AMD ✅ Intel
#[derive(PartialEq, Eq)]
#[repr(C)]
pub struct VendorInfo {
    ebx: u32,
    edx: u32,
    ecx: u32,
}

impl VendorInfo {
    /// Return vendor identification as human readable string.
    pub fn as_str(&self) -> &str {
        let brand_string_start = self as *const VendorInfo as *const u8;
        let slice = unsafe {
            // Safety: VendorInfo is laid out with repr(C) and exactly
            // 12 byte long without any padding.
            slice::from_raw_parts(brand_string_start, size_of::<VendorInfo>())
        };

        str::from_utf8(slice).unwrap_or("InvalidVendorString")
    }

    #[deprecated(
        since = "10.0.0",
        note = "Use idiomatic function name `as_str` instead"
    )]
    pub fn as_string(&self) -> &str {
        self.as_str()
    }
}

impl Debug for VendorInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VendorInfo")
            .field("brand_string", &self.as_str())
            .finish()
    }
}

impl fmt::Display for VendorInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Iterates over cache information (LEAF=0x02).
///
/// This will just return an index into a static table of cache descriptions
/// (see [CACHE_INFO_TABLE](crate::CACHE_INFO_TABLE)).
///
/// # Platforms
/// ❌ AMD ✅ Intel
#[derive(PartialEq, Eq, Clone)]
pub struct CacheInfoIter {
    current: u32,
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl Iterator for CacheInfoIter {
    type Item = CacheInfo;

    /// Iterate over all cache information.
    fn next(&mut self) -> Option<CacheInfo> {
        // Every byte of the 4 register values returned by cpuid
        // can contain information about a cache (except the
        // very first one).
        if self.current >= 4 * 4 {
            return None;
        }
        let reg_index = self.current % 4;
        let byte_index = self.current / 4;

        let reg = match reg_index {
            0 => self.eax,
            1 => self.ebx,
            2 => self.ecx,
            3 => self.edx,
            _ => unreachable!(),
        };

        let byte = match byte_index {
            0 => reg,
            1 => reg >> 8,
            2 => reg >> 16,
            3 => reg >> 24,
            _ => unreachable!(),
        } as u8;

        if byte == 0 {
            self.current += 1;
            return self.next();
        }

        for cache_info in CACHE_INFO_TABLE.iter() {
            if cache_info.num == byte {
                self.current += 1;
                return Some(*cache_info);
            }
        }

        None
    }
}

impl Debug for CacheInfoIter {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

/// What type of cache are we dealing with?
#[derive(Copy, Clone, Debug)]
pub enum CacheInfoType {
    General,
    Cache,
    TLB,
    STLB,
    DTLB,
    Prefetch,
}

/// Describes any kind of cache (TLB, Data and Instruction caches plus prefetchers).
#[derive(Copy, Clone)]
pub struct CacheInfo {
    /// Number as retrieved from cpuid
    pub num: u8,
    /// Cache type
    pub typ: CacheInfoType,
}

impl CacheInfo {
    /// Description of the cache (from Intel Manual)
    pub fn desc(&self) -> &'static str {
        match self.num {
            0x00 => "Null descriptor, this byte contains no information",
            0x01 => "Instruction TLB: 4 KByte pages, 4-way set associative, 32 entries",
            0x02 => "Instruction TLB: 4 MByte pages, fully associative, 2 entries",
            0x03 => "Data TLB: 4 KByte pages, 4-way set associative, 64 entries",
            0x04 => "Data TLB: 4 MByte pages, 4-way set associative, 8 entries",
            0x05 => "Data TLB1: 4 MByte pages, 4-way set associative, 32 entries",
            0x06 => "1st-level instruction cache: 8 KBytes, 4-way set associative, 32 byte line size",
            0x08 => "1st-level instruction cache: 16 KBytes, 4-way set associative, 32 byte line size",
            0x09 => "1st-level instruction cache: 32KBytes, 4-way set associative, 64 byte line size",
            0x0A => "1st-level data cache: 8 KBytes, 2-way set associative, 32 byte line size",
            0x0B => "Instruction TLB: 4 MByte pages, 4-way set associative, 4 entries",
            0x0C => "1st-level data cache: 16 KBytes, 4-way set associative, 32 byte line size",
            0x0D => "1st-level data cache: 16 KBytes, 4-way set associative, 64 byte line size",
            0x0E => "1st-level data cache: 24 KBytes, 6-way set associative, 64 byte line size",
            0x1D => "2nd-level cache: 128 KBytes, 2-way set associative, 64 byte line size",
            0x21 => "2nd-level cache: 256 KBytes, 8-way set associative, 64 byte line size",
            0x22 => "3rd-level cache: 512 KBytes, 4-way set associative, 64 byte line size, 2 lines per sector",
            0x23 => "3rd-level cache: 1 MBytes, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x24 => "2nd-level cache: 1 MBytes, 16-way set associative, 64 byte line size",
            0x25 => "3rd-level cache: 2 MBytes, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x29 => "3rd-level cache: 4 MBytes, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x2C => "1st-level data cache: 32 KBytes, 8-way set associative, 64 byte line size",
            0x30 => "1st-level instruction cache: 32 KBytes, 8-way set associative, 64 byte line size",
            0x40 => "No 2nd-level cache or, if processor contains a valid 2nd-level cache, no 3rd-level cache",
            0x41 => "2nd-level cache: 128 KBytes, 4-way set associative, 32 byte line size",
            0x42 => "2nd-level cache: 256 KBytes, 4-way set associative, 32 byte line size",
            0x43 => "2nd-level cache: 512 KBytes, 4-way set associative, 32 byte line size",
            0x44 => "2nd-level cache: 1 MByte, 4-way set associative, 32 byte line size",
            0x45 => "2nd-level cache: 2 MByte, 4-way set associative, 32 byte line size",
            0x46 => "3rd-level cache: 4 MByte, 4-way set associative, 64 byte line size",
            0x47 => "3rd-level cache: 8 MByte, 8-way set associative, 64 byte line size",
            0x48 => "2nd-level cache: 3MByte, 12-way set associative, 64 byte line size",
            0x49 => "3rd-level cache: 4MB, 16-way set associative, 64-byte line size (Intel Xeon processor MP, Family 0FH, Model 06H); 2nd-level cache: 4 MByte, 16-way set ssociative, 64 byte line size",
            0x4A => "3rd-level cache: 6MByte, 12-way set associative, 64 byte line size",
            0x4B => "3rd-level cache: 8MByte, 16-way set associative, 64 byte line size",
            0x4C => "3rd-level cache: 12MByte, 12-way set associative, 64 byte line size",
            0x4D => "3rd-level cache: 16MByte, 16-way set associative, 64 byte line size",
            0x4E => "2nd-level cache: 6MByte, 24-way set associative, 64 byte line size",
            0x4F => "Instruction TLB: 4 KByte pages, 32 entries",
            0x50 => "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 64 entries",
            0x51 => "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 128 entries",
            0x52 => "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 256 entries",
            0x55 => "Instruction TLB: 2-MByte or 4-MByte pages, fully associative, 7 entries",
            0x56 => "Data TLB0: 4 MByte pages, 4-way set associative, 16 entries",
            0x57 => "Data TLB0: 4 KByte pages, 4-way associative, 16 entries",
            0x59 => "Data TLB0: 4 KByte pages, fully associative, 16 entries",
            0x5A => "Data TLB0: 2-MByte or 4 MByte pages, 4-way set associative, 32 entries",
            0x5B => "Data TLB: 4 KByte and 4 MByte pages, 64 entries",
            0x5C => "Data TLB: 4 KByte and 4 MByte pages,128 entries",
            0x5D => "Data TLB: 4 KByte and 4 MByte pages,256 entries",
            0x60 => "1st-level data cache: 16 KByte, 8-way set associative, 64 byte line size",
            0x61 => "Instruction TLB: 4 KByte pages, fully associative, 48 entries",
            0x63 => "Data TLB: 2 MByte or 4 MByte pages, 4-way set associative, 32 entries and a separate array with 1 GByte pages, 4-way set associative, 4 entries",
            0x64 => "Data TLB: 4 KByte pages, 4-way set associative, 512 entries",
            0x66 => "1st-level data cache: 8 KByte, 4-way set associative, 64 byte line size",
            0x67 => "1st-level data cache: 16 KByte, 4-way set associative, 64 byte line size",
            0x68 => "1st-level data cache: 32 KByte, 4-way set associative, 64 byte line size",
            0x6A => "uTLB: 4 KByte pages, 8-way set associative, 64 entries",
            0x6B => "DTLB: 4 KByte pages, 8-way set associative, 256 entries",
            0x6C => "DTLB: 2M/4M pages, 8-way set associative, 128 entries",
            0x6D => "DTLB: 1 GByte pages, fully associative, 16 entries",
            0x70 => "Trace cache: 12 K-μop, 8-way set associative",
            0x71 => "Trace cache: 16 K-μop, 8-way set associative",
            0x72 => "Trace cache: 32 K-μop, 8-way set associative",
            0x76 => "Instruction TLB: 2M/4M pages, fully associative, 8 entries",
            0x78 => "2nd-level cache: 1 MByte, 4-way set associative, 64byte line size",
            0x79 => "2nd-level cache: 128 KByte, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x7A => "2nd-level cache: 256 KByte, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x7B => "2nd-level cache: 512 KByte, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x7C => "2nd-level cache: 1 MByte, 8-way set associative, 64 byte line size, 2 lines per sector",
            0x7D => "2nd-level cache: 2 MByte, 8-way set associative, 64byte line size",
            0x7F => "2nd-level cache: 512 KByte, 2-way set associative, 64-byte line size",
            0x80 => "2nd-level cache: 512 KByte, 8-way set associative, 64-byte line size",
            0x82 => "2nd-level cache: 256 KByte, 8-way set associative, 32 byte line size",
            0x83 => "2nd-level cache: 512 KByte, 8-way set associative, 32 byte line size",
            0x84 => "2nd-level cache: 1 MByte, 8-way set associative, 32 byte line size",
            0x85 => "2nd-level cache: 2 MByte, 8-way set associative, 32 byte line size",
            0x86 => "2nd-level cache: 512 KByte, 4-way set associative, 64 byte line size",
            0x87 => "2nd-level cache: 1 MByte, 8-way set associative, 64 byte line size",
            0xA0 => "DTLB: 4k pages, fully associative, 32 entries",
            0xB0 => "Instruction TLB: 4 KByte pages, 4-way set associative, 128 entries",
            0xB1 => "Instruction TLB: 2M pages, 4-way, 8 entries or 4M pages, 4-way, 4 entries",
            0xB2 => "Instruction TLB: 4KByte pages, 4-way set associative, 64 entries",
            0xB3 => "Data TLB: 4 KByte pages, 4-way set associative, 128 entries",
            0xB4 => "Data TLB1: 4 KByte pages, 4-way associative, 256 entries",
            0xB5 => "Instruction TLB: 4KByte pages, 8-way set associative, 64 entries",
            0xB6 => "Instruction TLB: 4KByte pages, 8-way set associative, 128 entries",
            0xBA => "Data TLB1: 4 KByte pages, 4-way associative, 64 entries",
            0xC0 => "Data TLB: 4 KByte and 4 MByte pages, 4-way associative, 8 entries",
            0xC1 => "Shared 2nd-Level TLB: 4 KByte/2MByte pages, 8-way associative, 1024 entries",
            0xC2 => "DTLB: 2 MByte/$MByte pages, 4-way associative, 16 entries",
            0xC3 => "Shared 2nd-Level TLB: 4 KByte /2 MByte pages, 6-way associative, 1536 entries. Also 1GBbyte pages, 4-way, 16 entries.",
            0xC4 => "DTLB: 2M/4M Byte pages, 4-way associative, 32 entries",
            0xCA => "Shared 2nd-Level TLB: 4 KByte pages, 4-way associative, 512 entries",
            0xD0 => "3rd-level cache: 512 KByte, 4-way set associative, 64 byte line size",
            0xD1 => "3rd-level cache: 1 MByte, 4-way set associative, 64 byte line size",
            0xD2 => "3rd-level cache: 2 MByte, 4-way set associative, 64 byte line size",
            0xD6 => "3rd-level cache: 1 MByte, 8-way set associative, 64 byte line size",
            0xD7 => "3rd-level cache: 2 MByte, 8-way set associative, 64 byte line size",
            0xD8 => "3rd-level cache: 4 MByte, 8-way set associative, 64 byte line size",
            0xDC => "3rd-level cache: 1.5 MByte, 12-way set associative, 64 byte line size",
            0xDD => "3rd-level cache: 3 MByte, 12-way set associative, 64 byte line size",
            0xDE => "3rd-level cache: 6 MByte, 12-way set associative, 64 byte line size",
            0xE2 => "3rd-level cache: 2 MByte, 16-way set associative, 64 byte line size",
            0xE3 => "3rd-level cache: 4 MByte, 16-way set associative, 64 byte line size",
            0xE4 => "3rd-level cache: 8 MByte, 16-way set associative, 64 byte line size",
            0xEA => "3rd-level cache: 12MByte, 24-way set associative, 64 byte line size",
            0xEB => "3rd-level cache: 18MByte, 24-way set associative, 64 byte line size",
            0xEC => "3rd-level cache: 24MByte, 24-way set associative, 64 byte line size",
            0xF0 => "64-Byte prefetching",
            0xF1 => "128-Byte prefetching",
            0xFE => "CPUID leaf 2 does not report TLB descriptor information; use CPUID leaf 18H to query TLB and other address translation parameters.",
            0xFF => "CPUID leaf 2 does not report cache descriptor information, use CPUID leaf 4 to query cache parameters",
            _ => "Unknown cache type!"
        }
    }
}

impl Debug for CacheInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("CacheInfo")
            .field("typ", &self.typ)
            .field("desc", &self.desc())
            .finish()
    }
}

impl fmt::Display for CacheInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let typ = match self.typ {
            CacheInfoType::General => "N/A",
            CacheInfoType::Cache => "Cache",
            CacheInfoType::TLB => "TLB",
            CacheInfoType::STLB => "STLB",
            CacheInfoType::DTLB => "DTLB",
            CacheInfoType::Prefetch => "Prefetcher",
        };

        write!(f, "{:x}:\t {}: {}", self.num, typ, self.desc())
    }
}

/// This table is taken from Intel manual (Section CPUID instruction).
pub const CACHE_INFO_TABLE: [CacheInfo; 108] = [
    CacheInfo {
        num: 0x00,
        typ: CacheInfoType::General,
    },
    CacheInfo {
        num: 0x01,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x02,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x03,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x04,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x05,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x06,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x08,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x09,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x0A,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x0B,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x0C,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x0D,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x0E,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x21,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x22,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x23,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x24,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x25,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x29,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x2C,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x30,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x40,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x41,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x42,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x43,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x44,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x45,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x46,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x47,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x48,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x49,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4A,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4B,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4C,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4D,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4E,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x4F,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x50,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x51,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x52,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x55,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x56,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x57,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x59,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x5A,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x5B,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x5C,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x5D,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x60,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x61,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x63,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x66,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x67,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x68,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x6A,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x6B,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x6C,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x6D,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x70,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x71,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x72,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x76,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0x78,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x79,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x7A,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x7B,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x7C,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x7D,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x7F,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x80,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x82,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x83,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x84,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x85,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x86,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0x87,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xB0,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB1,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB2,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB3,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB4,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB5,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xB6,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xBA,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xC0,
        typ: CacheInfoType::TLB,
    },
    CacheInfo {
        num: 0xC1,
        typ: CacheInfoType::STLB,
    },
    CacheInfo {
        num: 0xC2,
        typ: CacheInfoType::DTLB,
    },
    CacheInfo {
        num: 0xCA,
        typ: CacheInfoType::STLB,
    },
    CacheInfo {
        num: 0xD0,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xD1,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xD2,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xD6,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xD7,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xD8,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xDC,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xDD,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xDE,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xE2,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xE3,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xE4,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xEA,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xEB,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xEC,
        typ: CacheInfoType::Cache,
    },
    CacheInfo {
        num: 0xF0,
        typ: CacheInfoType::Prefetch,
    },
    CacheInfo {
        num: 0xF1,
        typ: CacheInfoType::Prefetch,
    },
    CacheInfo {
        num: 0xFE,
        typ: CacheInfoType::General,
    },
    CacheInfo {
        num: 0xFF,
        typ: CacheInfoType::General,
    },
];

/// Processor Serial Number (LEAF=0x3).
///
/// # Deprecated
///
/// Processor serial number (PSN) is not supported in the Pentium 4 processor or
/// later. On all models, use the PSN flag (returned using CPUID) to check for
/// PSN support before accessing the feature.
///
/// # Platforms
/// ❌ AMD ✅ Intel
#[derive(PartialEq, Eq)]
pub struct ProcessorSerial {
    /// Lower bits
    ecx: u32,
    /// Middle bits
    edx: u32,
    /// Upper bits (come from leaf 0x1)
    eax: u32,
}

impl ProcessorSerial {
    /// Bits 00-31 of 96 bit processor serial number.
    ///
    /// (Available in Pentium III processor only; otherwise, the value in this register is reserved.)
    pub fn serial_lower(&self) -> u32 {
        self.ecx
    }

    /// Bits 32-63 of 96 bit processor serial number.
    ///
    /// (Available in Pentium III processor only; otherwise, the value in this register is reserved.)
    pub fn serial_middle(&self) -> u32 {
        self.edx
    }

    /// Bits 64-96 of 96 bit processor serial number.
    pub fn serial_upper(&self) -> u32 {
        self.eax
    }

    /// Combination of bits 00-31 and 32-63 of 96 bit processor serial number.
    pub fn serial(&self) -> u64 {
        (self.serial_lower() as u64) | (self.serial_middle() as u64) << 32
    }

    /// 96 bit processor serial number.
    pub fn serial_all(&self) -> u128 {
        (self.serial_lower() as u128)
            | ((self.serial_middle() as u128) << 32)
            | ((self.serial_upper() as u128) << 64)
    }
}

impl Debug for ProcessorSerial {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProcessorSerial")
            .field("serial_lower", &self.serial_lower())
            .field("serial_middle", &self.serial_middle())
            .finish()
    }
}

/// Processor and Processor Feature Identifiers (LEAF=0x01).
///
/// # Platforms
/// ✅ AMD ✅ Intel
pub struct FeatureInfo {
    vendor: Vendor,
    eax: u32,
    ebx: u32,
    edx_ecx: FeatureInfoFlags,
}

impl FeatureInfo {
    /// Version Information: Extended Family
    pub fn extended_family_id(&self) -> u8 {
        get_bits(self.eax, 20, 27) as u8
    }

    /// Version Information: Extended Model
    pub fn extended_model_id(&self) -> u8 {
        get_bits(self.eax, 16, 19) as u8
    }

    /// Version Information: Family
    pub fn base_family_id(&self) -> u8 {
        get_bits(self.eax, 8, 11) as u8
    }

    /// Version Information: Model
    pub fn base_model_id(&self) -> u8 {
        get_bits(self.eax, 4, 7) as u8
    }

    pub fn family_id(&self) -> u8 {
        let base_family_id = self.base_family_id();
        let extended_family_id = self.extended_family_id();
        let just_use_base = (self.vendor == Vendor::Amd && base_family_id < 0xf)
            || (self.vendor == Vendor::Intel && base_family_id != 0xf);

        if just_use_base {
            base_family_id
        } else {
            base_family_id + extended_family_id
        }
    }

    pub fn model_id(&self) -> u8 {
        let base_family_id = self.base_family_id();
        let base_model_id = self.base_model_id();
        let extended_model_id = self.extended_model_id();
        let just_use_base = (self.vendor == Vendor::Amd && base_family_id < 0xf)
            || (self.vendor == Vendor::Intel && base_family_id != 0xf && base_family_id != 0x6);

        if just_use_base {
            base_model_id
        } else {
            (extended_model_id << 4) | base_model_id
        }
    }

    /// Version Information: Stepping ID
    pub fn stepping_id(&self) -> u8 {
        get_bits(self.eax, 0, 3) as u8
    }

    /// Brand Index
    pub fn brand_index(&self) -> u8 {
        get_bits(self.ebx, 0, 7) as u8
    }

    /// CLFLUSH line size (Value ∗ 8 = cache line size in bytes)
    pub fn cflush_cache_line_size(&self) -> u8 {
        get_bits(self.ebx, 8, 15) as u8
    }

    /// Initial APIC ID
    pub fn initial_local_apic_id(&self) -> u8 {
        get_bits(self.ebx, 24, 31) as u8
    }

    /// Maximum number of addressable IDs for logical processors in this physical package.
    pub fn max_logical_processor_ids(&self) -> u8 {
        get_bits(self.ebx, 16, 23) as u8
    }

    check_flag!(
        doc = "Streaming SIMD Extensions 3 (SSE3). A value of 1 indicates the processor \
               supports this technology.",
        has_sse3,
        edx_ecx,
        FeatureInfoFlags::SSE3
    );

    check_flag!(
        doc = "PCLMULQDQ. A value of 1 indicates the processor supports the PCLMULQDQ \
               instruction",
        has_pclmulqdq,
        edx_ecx,
        FeatureInfoFlags::PCLMULQDQ
    );

    check_flag!(
        doc = "64-bit DS Area. A value of 1 indicates the processor supports DS area \
               using 64-bit layout",
        has_ds_area,
        edx_ecx,
        FeatureInfoFlags::DTES64
    );

    check_flag!(
        doc = "MONITOR/MWAIT. A value of 1 indicates the processor supports this feature.",
        has_monitor_mwait,
        edx_ecx,
        FeatureInfoFlags::MONITOR
    );

    check_flag!(
        doc = "CPL Qualified Debug Store. A value of 1 indicates the processor supports \
               the extensions to the  Debug Store feature to allow for branch message \
               storage qualified by CPL.",
        has_cpl,
        edx_ecx,
        FeatureInfoFlags::DSCPL
    );

    check_flag!(
        doc = "Virtual Machine Extensions. A value of 1 indicates that the processor \
               supports this technology.",
        has_vmx,
        edx_ecx,
        FeatureInfoFlags::VMX
    );

    check_flag!(
        doc = "Safer Mode Extensions. A value of 1 indicates that the processor supports \
               this technology. See Chapter 5, Safer Mode Extensions Reference.",
        has_smx,
        edx_ecx,
        FeatureInfoFlags::SMX
    );

    check_flag!(
        doc = "Enhanced Intel SpeedStep® technology. A value of 1 indicates that the \
               processor supports this technology.",
        has_eist,
        edx_ecx,
        FeatureInfoFlags::EIST
    );

    check_flag!(
        doc = "Thermal Monitor 2. A value of 1 indicates whether the processor supports \
               this technology.",
        has_tm2,
        edx_ecx,
        FeatureInfoFlags::TM2
    );

    check_flag!(
        doc = "A value of 1 indicates the presence of the Supplemental Streaming SIMD \
               Extensions 3 (SSSE3). A value of 0 indicates the instruction extensions \
               are not present in the processor",
        has_ssse3,
        edx_ecx,
        FeatureInfoFlags::SSSE3
    );

    check_flag!(
        doc = "L1 Context ID. A value of 1 indicates the L1 data cache mode can be set \
               to either adaptive mode or shared mode. A value of 0 indicates this \
               feature is not supported. See definition of the IA32_MISC_ENABLE MSR Bit \
               24 (L1 Data Cache Context Mode) for details.",
        has_cnxtid,
        edx_ecx,
        FeatureInfoFlags::CNXTID
    );

    check_flag!(
        doc = "A value of 1 indicates the processor supports FMA extensions using YMM \
               state.",
        has_fma,
        edx_ecx,
        FeatureInfoFlags::FMA
    );

    check_flag!(
        doc = "CMPXCHG16B Available. A value of 1 indicates that the feature is \
               available. See the CMPXCHG8B/CMPXCHG16B Compare and Exchange Bytes \
               section. 14",
        has_cmpxchg16b,
        edx_ecx,
        FeatureInfoFlags::CMPXCHG16B
    );

    check_flag!(
        doc = "Perfmon and Debug Capability: A value of 1 indicates the processor \
               supports the performance   and debug feature indication MSR \
               IA32_PERF_CAPABILITIES.",
        has_pdcm,
        edx_ecx,
        FeatureInfoFlags::PDCM
    );

    check_flag!(
        doc = "Process-context identifiers. A value of 1 indicates that the processor \
               supports PCIDs and the software may set CR4.PCIDE to 1.",
        has_pcid,
        edx_ecx,
        FeatureInfoFlags::PCID
    );

    check_flag!(
        doc = "A value of 1 indicates the processor supports the ability to prefetch \
               data from a memory mapped device.",
        has_dca,
        edx_ecx,
        FeatureInfoFlags::DCA
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports SSE4.1.",
        has_sse41,
        edx_ecx,
        FeatureInfoFlags::SSE41
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports SSE4.2.",
        has_sse42,
        edx_ecx,
        FeatureInfoFlags::SSE42
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports x2APIC feature.",
        has_x2apic,
        edx_ecx,
        FeatureInfoFlags::X2APIC
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports MOVBE instruction.",
        has_movbe,
        edx_ecx,
        FeatureInfoFlags::MOVBE
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports the POPCNT instruction.",
        has_popcnt,
        edx_ecx,
        FeatureInfoFlags::POPCNT
    );

    check_flag!(
        doc = "A value of 1 indicates that the processors local APIC timer supports \
               one-shot operation using a TSC deadline value.",
        has_tsc_deadline,
        edx_ecx,
        FeatureInfoFlags::TSC_DEADLINE
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports the AESNI instruction \
               extensions.",
        has_aesni,
        edx_ecx,
        FeatureInfoFlags::AESNI
    );

    check_flag!(
        doc = "A value of 1 indicates that the processor supports the XSAVE/XRSTOR \
               processor extended states feature, the XSETBV/XGETBV instructions, and \
               XCR0.",
        has_xsave,
        edx_ecx,
        FeatureInfoFlags::XSAVE
    );

    check_flag!(
        doc = "A value of 1 indicates that the OS has enabled XSETBV/XGETBV instructions \
               to access XCR0, and support for processor extended state management using \
               XSAVE/XRSTOR.",
        has_oxsave,
        edx_ecx,
        FeatureInfoFlags::OSXSAVE
    );

    check_flag!(
        doc = "A value of 1 indicates the processor supports the AVX instruction \
               extensions.",
        has_avx,
        edx_ecx,
        FeatureInfoFlags::AVX
    );

    check_flag!(
        doc = "A value of 1 indicates that processor supports 16-bit floating-point \
               conversion instructions.",
        has_f16c,
        edx_ecx,
        FeatureInfoFlags::F16C
    );

    check_flag!(
        doc = "A value of 1 indicates that processor supports RDRAND instruction.",
        has_rdrand,
        edx_ecx,
        FeatureInfoFlags::RDRAND
    );

    check_flag!(
        doc = "A value of 1 indicates the indicates the presence of a hypervisor.",
        has_hypervisor,
        edx_ecx,
        FeatureInfoFlags::HYPERVISOR
    );

    check_flag!(
        doc = "Floating Point Unit On-Chip. The processor contains an x87 FPU.",
        has_fpu,
        edx_ecx,
        FeatureInfoFlags::FPU
    );

    check_flag!(
        doc = "Virtual 8086 Mode Enhancements. Virtual 8086 mode enhancements, including \
               CR4.VME for controlling the feature, CR4.PVI for protected mode virtual \
               interrupts, software interrupt indirection, expansion of the TSS with the \
               software indirection bitmap, and EFLAGS.VIF and EFLAGS.VIP flags.",
        has_vme,
        edx_ecx,
        FeatureInfoFlags::VME
    );

    check_flag!(
        doc = "Debugging Extensions. Support for I/O breakpoints, including CR4.DE for \
               controlling the feature, and optional trapping of accesses to DR4 and DR5.",
        has_de,
        edx_ecx,
        FeatureInfoFlags::DE
    );

    check_flag!(
        doc = "Page Size Extension. Large pages of size 4 MByte are supported, including \
               CR4.PSE for controlling the feature, the defined dirty bit in PDE (Page \
               Directory Entries), optional reserved bit trapping in CR3, PDEs, and PTEs.",
        has_pse,
        edx_ecx,
        FeatureInfoFlags::PSE
    );

    check_flag!(
        doc = "Time Stamp Counter. The RDTSC instruction is supported, including CR4.TSD \
               for controlling privilege.",
        has_tsc,
        edx_ecx,
        FeatureInfoFlags::TSC
    );

    check_flag!(
        doc = "Model Specific Registers RDMSR and WRMSR Instructions. The RDMSR and \
               WRMSR instructions are supported. Some of the MSRs are implementation \
               dependent.",
        has_msr,
        edx_ecx,
        FeatureInfoFlags::MSR
    );

    check_flag!(
        doc = "Physical Address Extension. Physical addresses greater than 32 bits are \
               supported: extended page table entry formats, an extra level in the page \
               translation tables is defined, 2-MByte pages are supported instead of 4 \
               Mbyte pages if PAE bit is 1.",
        has_pae,
        edx_ecx,
        FeatureInfoFlags::PAE
    );

    check_flag!(
        doc = "Machine Check Exception. Exception 18 is defined for Machine Checks, \
               including CR4.MCE for controlling the feature. This feature does not \
               define the model-specific implementations of machine-check error logging, \
               reporting, and processor shutdowns. Machine Check exception handlers may \
               have to depend on processor version to do model specific processing of \
               the exception, or test for the presence of the Machine Check feature.",
        has_mce,
        edx_ecx,
        FeatureInfoFlags::MCE
    );

    check_flag!(
        doc = "CMPXCHG8B Instruction. The compare-and-exchange 8 bytes (64 bits) \
               instruction is supported (implicitly locked and atomic).",
        has_cmpxchg8b,
        edx_ecx,
        FeatureInfoFlags::CX8
    );

    check_flag!(
        doc = "APIC On-Chip. The processor contains an Advanced Programmable Interrupt \
               Controller (APIC), responding to memory mapped commands in the physical \
               address range FFFE0000H to FFFE0FFFH (by default - some processors permit \
               the APIC to be relocated).",
        has_apic,
        edx_ecx,
        FeatureInfoFlags::APIC
    );

    check_flag!(
        doc = "SYSENTER and SYSEXIT Instructions. The SYSENTER and SYSEXIT and \
               associated MSRs are supported.",
        has_sysenter_sysexit,
        edx_ecx,
        FeatureInfoFlags::SEP
    );

    check_flag!(
        doc = "Memory Type Range Registers. MTRRs are supported. The MTRRcap MSR \
               contains feature bits that describe what memory types are supported, how \
               many variable MTRRs are supported, and whether fixed MTRRs are supported.",
        has_mtrr,
        edx_ecx,
        FeatureInfoFlags::MTRR
    );

    check_flag!(
        doc = "Page Global Bit. The global bit is supported in paging-structure entries \
               that map a page, indicating TLB entries that are common to different \
               processes and need not be flushed. The CR4.PGE bit controls this feature.",
        has_pge,
        edx_ecx,
        FeatureInfoFlags::PGE
    );

    check_flag!(
        doc = "Machine Check Architecture. A value of 1 indicates the Machine Check \
               Architecture of reporting machine errors is supported. The MCG_CAP MSR \
               contains feature bits describing how many banks of error reporting MSRs \
               are supported.",
        has_mca,
        edx_ecx,
        FeatureInfoFlags::MCA
    );

    check_flag!(
        doc = "Conditional Move Instructions. The conditional move instruction CMOV is \
               supported. In addition, if x87 FPU is present as indicated by the \
               CPUID.FPU feature bit, then the FCOMI and FCMOV instructions are supported",
        has_cmov,
        edx_ecx,
        FeatureInfoFlags::CMOV
    );

    check_flag!(
        doc = "Page Attribute Table. Page Attribute Table is supported. This feature \
               augments the Memory Type Range Registers (MTRRs), allowing an operating \
               system to specify attributes of memory accessed through a linear address \
               on a 4KB granularity.",
        has_pat,
        edx_ecx,
        FeatureInfoFlags::PAT
    );

    check_flag!(
        doc = "36-Bit Page Size Extension. 4-MByte pages addressing physical memory \
               beyond 4 GBytes are supported with 32-bit paging. This feature indicates \
               that upper bits of the physical address of a 4-MByte page are encoded in \
               bits 20:13 of the page-directory entry. Such physical addresses are \
               limited by MAXPHYADDR and may be up to 40 bits in size.",
        has_pse36,
        edx_ecx,
        FeatureInfoFlags::PSE36
    );

    check_flag!(
        doc = "Processor Serial Number. The processor supports the 96-bit processor \
               identification number feature and the feature is enabled.",
        has_psn,
        edx_ecx,
        FeatureInfoFlags::PSN
    );

    check_flag!(
        doc = "CLFLUSH Instruction. CLFLUSH Instruction is supported.",
        has_clflush,
        edx_ecx,
        FeatureInfoFlags::CLFSH
    );

    check_flag!(
        doc = "Debug Store. The processor supports the ability to write debug \
               information into a memory resident buffer. This feature is used by the \
               branch trace store (BTS) and processor event-based sampling (PEBS) \
               facilities (see Chapter 23, Introduction to Virtual-Machine Extensions, \
               in the Intel® 64 and IA-32 Architectures Software Developers Manual, \
               Volume 3C).",
        has_ds,
        edx_ecx,
        FeatureInfoFlags::DS
    );

    check_flag!(
        doc = "Thermal Monitor and Software Controlled Clock Facilities. The processor \
               implements internal MSRs that allow processor temperature to be monitored \
               and processor performance to be modulated in predefined duty cycles under \
               software control.",
        has_acpi,
        edx_ecx,
        FeatureInfoFlags::ACPI
    );

    check_flag!(
        doc = "Intel MMX Technology. The processor supports the Intel MMX technology.",
        has_mmx,
        edx_ecx,
        FeatureInfoFlags::MMX
    );

    check_flag!(
        doc = "FXSAVE and FXRSTOR Instructions. The FXSAVE and FXRSTOR instructions are \
               supported for fast save and restore of the floating point context. \
               Presence of this bit also indicates that CR4.OSFXSR is available for an \
               operating system to indicate that it supports the FXSAVE and FXRSTOR \
               instructions.",
        has_fxsave_fxstor,
        edx_ecx,
        FeatureInfoFlags::FXSR
    );

    check_flag!(
        doc = "SSE. The processor supports the SSE extensions.",
        has_sse,
        edx_ecx,
        FeatureInfoFlags::SSE
    );

    check_flag!(
        doc = "SSE2. The processor supports the SSE2 extensions.",
        has_sse2,
        edx_ecx,
        FeatureInfoFlags::SSE2
    );

    check_flag!(
        doc = "Self Snoop. The processor supports the management of conflicting memory \
               types by performing a snoop of its own cache structure for transactions \
               issued to the bus.",
        has_ss,
        edx_ecx,
        FeatureInfoFlags::SS
    );

    check_flag!(
        doc = "Max APIC IDs reserved field is Valid. A value of 0 for HTT indicates \
               there is only a single logical processor in the package and software \
               should assume only a single APIC ID is reserved.  A value of 1 for HTT \
               indicates the value in CPUID.1.EBX\\[23:16\\] (the Maximum number of \
               addressable IDs for logical processors in this package) is valid for the \
               package.",
        has_htt,
        edx_ecx,
        FeatureInfoFlags::HTT
    );

    check_flag!(
        doc = "Thermal Monitor. The processor implements the thermal monitor automatic \
               thermal control circuitry (TCC).",
        has_tm,
        edx_ecx,
        FeatureInfoFlags::TM
    );

    check_flag!(
        doc = "Pending Break Enable. The processor supports the use of the FERR#/PBE# \
               pin when the processor is in the stop-clock state (STPCLK# is asserted) \
               to signal the processor that an interrupt is pending and that the \
               processor should return to normal operation to handle the interrupt. Bit \
               10 (PBE enable) in the IA32_MISC_ENABLE MSR enables this capability.",
        has_pbe,
        edx_ecx,
        FeatureInfoFlags::PBE
    );
}

impl Debug for FeatureInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FeatureInfo")
            .field("extended_family_id", &self.extended_family_id())
            .field("extended_model_id", &self.extended_model_id())
            .field("family_id", &self.family_id())
            .field("model_id", &self.model_id())
            .field("stepping_id", &self.stepping_id())
            .field("brand_index", &self.brand_index())
            .field("cflush_cache_line_size", &self.cflush_cache_line_size())
            .field("initial_local_apic_id", &self.initial_local_apic_id())
            .field(
                "max_logical_processor_ids",
                &self.max_logical_processor_ids(),
            )
            .field("edx_ecx", &self.edx_ecx)
            .finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FeatureInfoFlags: u64 {
        // ECX flags

        /// Streaming SIMD Extensions 3 (SSE3). A value of 1 indicates the processor supports this technology.
        const SSE3 = 1 << 0;
        /// PCLMULQDQ. A value of 1 indicates the processor supports the PCLMULQDQ instruction
        const PCLMULQDQ = 1 << 1;
        /// 64-bit DS Area. A value of 1 indicates the processor supports DS area using 64-bit layout
        const DTES64 = 1 << 2;
        /// MONITOR/MWAIT. A value of 1 indicates the processor supports this feature.
        const MONITOR = 1 << 3;
        /// CPL Qualified Debug Store. A value of 1 indicates the processor supports the extensions to the  Debug Store feature to allow for branch message storage qualified by CPL.
        const DSCPL = 1 << 4;
        /// Virtual Machine Extensions. A value of 1 indicates that the processor supports this technology.
        const VMX = 1 << 5;
        /// Safer Mode Extensions. A value of 1 indicates that the processor supports this technology. See Chapter 5, Safer Mode Extensions Reference.
        const SMX = 1 << 6;
        /// Enhanced Intel SpeedStep® technology. A value of 1 indicates that the processor supports this technology.
        const EIST = 1 << 7;
        /// Thermal Monitor 2. A value of 1 indicates whether the processor supports this technology.
        const TM2 = 1 << 8;
        /// A value of 1 indicates the presence of the Supplemental Streaming SIMD Extensions 3 (SSSE3). A value of 0 indicates the instruction extensions are not present in the processor
        const SSSE3 = 1 << 9;
        /// L1 Context ID. A value of 1 indicates the L1 data cache mode can be set to either adaptive mode or shared mode. A value of 0 indicates this feature is not supported. See definition of the IA32_MISC_ENABLE MSR Bit 24 (L1 Data Cache Context Mode) for details.
        const CNXTID = 1 << 10;
        /// A value of 1 indicates the processor supports FMA extensions using YMM state.
        const FMA = 1 << 12;
        /// CMPXCHG16B Available. A value of 1 indicates that the feature is available. See the CMPXCHG8B/CMPXCHG16B Compare and Exchange Bytes section. 14
        const CMPXCHG16B = 1 << 13;
        /// Perfmon and Debug Capability: A value of 1 indicates the processor supports the performance   and debug feature indication MSR IA32_PERF_CAPABILITIES.
        const PDCM = 1 << 15;
        /// Process-context identifiers. A value of 1 indicates that the processor supports PCIDs and the software may set CR4.PCIDE to 1.
        const PCID = 1 << 17;
        /// A value of 1 indicates the processor supports the ability to prefetch data from a memory mapped device.
        const DCA = 1 << 18;
        /// A value of 1 indicates that the processor supports SSE4.1.
        const SSE41 = 1 << 19;
        /// A value of 1 indicates that the processor supports SSE4.2.
        const SSE42 = 1 << 20;
        /// A value of 1 indicates that the processor supports x2APIC feature.
        const X2APIC = 1 << 21;
        /// A value of 1 indicates that the processor supports MOVBE instruction.
        const MOVBE = 1 << 22;
        /// A value of 1 indicates that the processor supports the POPCNT instruction.
        const POPCNT = 1 << 23;
        /// A value of 1 indicates that the processors local APIC timer supports one-shot operation using a TSC deadline value.
        const TSC_DEADLINE = 1 << 24;
        /// A value of 1 indicates that the processor supports the AESNI instruction extensions.
        const AESNI = 1 << 25;
        /// A value of 1 indicates that the processor supports the XSAVE/XRSTOR processor extended states feature, the XSETBV/XGETBV instructions, and XCR0.
        const XSAVE = 1 << 26;
        /// A value of 1 indicates that the OS has enabled XSETBV/XGETBV instructions to access XCR0, and support for processor extended state management using XSAVE/XRSTOR.
        const OSXSAVE = 1 << 27;
        /// A value of 1 indicates the processor supports the AVX instruction extensions.
        const AVX = 1 << 28;
        /// A value of 1 indicates that processor supports 16-bit floating-point conversion instructions.
        const F16C = 1 << 29;
        /// A value of 1 indicates that processor supports RDRAND instruction.
        const RDRAND = 1 << 30;
        /// A value of 1 indicates the indicates the presence of a hypervisor.
        const HYPERVISOR = 1 << 31;


        // EDX flags

        /// Floating Point Unit On-Chip. The processor contains an x87 FPU.
        const FPU = 1 << 32;
        /// Virtual 8086 Mode Enhancements. Virtual 8086 mode enhancements, including CR4.VME for controlling the feature, CR4.PVI for protected mode virtual interrupts, software interrupt indirection, expansion of the TSS with the software indirection bitmap, and EFLAGS.VIF and EFLAGS.VIP flags.
        const VME = 1 << (32 + 1);
        /// Debugging Extensions. Support for I/O breakpoints, including CR4.DE for controlling the feature, and optional trapping of accesses to DR4 and DR5.
        const DE = 1 << (32 + 2);
        /// Page Size Extension. Large pages of size 4 MByte are supported, including CR4.PSE for controlling the feature, the defined dirty bit in PDE (Page Directory Entries), optional reserved bit trapping in CR3, PDEs, and PTEs.
        const PSE = 1 << (32 + 3);
        /// Time Stamp Counter. The RDTSC instruction is supported, including CR4.TSD for controlling privilege.
        const TSC = 1 << (32 + 4);
        /// Model Specific Registers RDMSR and WRMSR Instructions. The RDMSR and WRMSR instructions are supported. Some of the MSRs are implementation dependent.
        const MSR = 1 << (32 + 5);
        /// Physical Address Extension. Physical addresses greater than 32 bits are supported: extended page table entry formats, an extra level in the page translation tables is defined, 2-MByte pages are supported instead of 4 Mbyte pages if PAE bit is 1.
        const PAE = 1 << (32 + 6);
        /// Machine Check Exception. Exception 18 is defined for Machine Checks, including CR4.MCE for controlling the feature. This feature does not define the model-specific implementations of machine-check error logging, reporting, and processor shutdowns. Machine Check exception handlers may have to depend on processor version to do model specific processing of the exception, or test for the presence of the Machine Check feature.
        const MCE = 1 << (32 + 7);
        /// CMPXCHG8B Instruction. The compare-and-exchange 8 bytes (64 bits) instruction is supported (implicitly locked and atomic).
        const CX8 = 1 << (32 + 8);
        /// APIC On-Chip. The processor contains an Advanced Programmable Interrupt Controller (APIC), responding to memory mapped commands in the physical address range FFFE0000H to FFFE0FFFH (by default - some processors permit the APIC to be relocated).
        const APIC = 1 << (32 + 9);
        /// SYSENTER and SYSEXIT Instructions. The SYSENTER and SYSEXIT and associated MSRs are supported.
        const SEP = 1 << (32 + 11);
        /// Memory Type Range Registers. MTRRs are supported. The MTRRcap MSR contains feature bits that describe what memory types are supported, how many variable MTRRs are supported, and whether fixed MTRRs are supported.
        const MTRR = 1 << (32 + 12);
        /// Page Global Bit. The global bit is supported in paging-structure entries that map a page, indicating TLB entries that are common to different processes and need not be flushed. The CR4.PGE bit controls this feature.
        const PGE = 1 << (32 + 13);
        /// Machine Check Architecture. The Machine Check exArchitecture, which provides a compatible mechanism for error reporting in P6 family, Pentium 4, Intel Xeon processors, and future processors, is supported. The MCG_CAP MSR contains feature bits describing how many banks of error reporting MSRs are supported.
        const MCA = 1 << (32 + 14);
        /// Conditional Move Instructions. The conditional move instruction CMOV is supported. In addition, if x87 FPU is present as indicated by the CPUID.FPU feature bit, then the FCOMI and FCMOV instructions are supported
        const CMOV = 1 << (32 + 15);
        /// Page Attribute Table. Page Attribute Table is supported. This feature augments the Memory Type Range Registers (MTRRs), allowing an operating system to specify attributes of memory accessed through a linear address on a 4KB granularity.
        const PAT = 1 << (32 + 16);
        /// 36-Bit Page Size Extension. 4-MByte pages addressing physical memory beyond 4 GBytes are supported with 32-bit paging. This feature indicates that upper bits of the physical address of a 4-MByte page are encoded in bits 20:13 of the page-directory entry. Such physical addresses are limited by MAXPHYADDR and may be up to 40 bits in size.
        const PSE36 = 1 << (32 + 17);
        /// Processor Serial Number. The processor supports the 96-bit processor identification number feature and the feature is enabled.
        const PSN = 1 << (32 + 18);
        /// CLFLUSH Instruction. CLFLUSH Instruction is supported.
        const CLFSH = 1 << (32 + 19);
        /// Debug Store. The processor supports the ability to write debug information into a memory resident buffer. This feature is used by the branch trace store (BTS) and precise event-based sampling (PEBS) facilities (see Chapter 23, Introduction to Virtual-Machine Extensions, in the Intel® 64 and IA-32 Architectures Software Developers Manual, Volume 3C).
        const DS = 1 << (32 + 21);
        /// Thermal Monitor and Software Controlled Clock Facilities. The processor implements internal MSRs that allow processor temperature to be monitored and processor performance to be modulated in predefined duty cycles under software control.
        const ACPI = 1 << (32 + 22);
        /// Intel MMX Technology. The processor supports the Intel MMX technology.
        const MMX = 1 << (32 + 23);
        /// FXSAVE and FXRSTOR Instructions. The FXSAVE and FXRSTOR instructions are supported for fast save and restore of the floating point context. Presence of this bit also indicates that CR4.OSFXSR is available for an operating system to indicate that it supports the FXSAVE and FXRSTOR instructions.
        const FXSR = 1 << (32 + 24);
        /// SSE. The processor supports the SSE extensions.
        const SSE = 1 << (32 + 25);
        /// SSE2. The processor supports the SSE2 extensions.
        const SSE2 = 1 << (32 + 26);
        /// Self Snoop. The processor supports the management of conflicting memory types by performing a snoop of its own cache structure for transactions issued to the bus.
        const SS = 1 << (32 + 27);
        /// Max APIC IDs reserved field is Valid. A value of 0 for HTT indicates there is only a single logical processor in the package and software should assume only a single APIC ID is reserved.  A value of 1 for HTT indicates the value in CPUID.1.EBX[23:16] (the Maximum number of addressable IDs for logical processors in this package) is valid for the package.
        const HTT = 1 << (32 + 28);
        /// Thermal Monitor. The processor implements the thermal monitor automatic thermal control circuitry (TCC).
        const TM = 1 << (32 + 29);
        /// Pending Break Enable. The processor supports the use of the FERR#/PBE# pin when the processor is in the stop-clock state (STPCLK# is asserted) to signal the processor that an interrupt is pending and that the processor should return to normal operation to handle the interrupt. Bit 10 (PBE enable) in the IA32_MISC_ENABLE MSR enables this capability.
        const PBE = 1 << (32 + 31);
    }
}

/// Iterator over caches (LEAF=0x04).
///
/// Yields a [CacheParameter] for each cache.
///
/// # Platforms
/// 🟡 AMD ✅ Intel
#[derive(Clone, Copy)]
pub struct CacheParametersIter<R: CpuIdReader> {
    read: R,
    leaf: u32,
    current: u32,
}

impl<R: CpuIdReader> Iterator for CacheParametersIter<R> {
    type Item = CacheParameter;

    /// Iterate over all cache info subleafs for this CPU.
    ///
    /// # Note
    /// cpuid is called every-time we advance the iterator to get information
    /// about the next cache.
    fn next(&mut self) -> Option<CacheParameter> {
        let res = self.read.cpuid2(self.leaf, self.current);
        let cp = CacheParameter {
            eax: res.eax,
            ebx: res.ebx,
            ecx: res.ecx,
            edx: res.edx,
        };

        match cp.cache_type() {
            CacheType::Null => None,
            CacheType::Reserved => None,
            _ => {
                self.current += 1;
                Some(cp)
            }
        }
    }
}

impl<R: CpuIdReader> Debug for CacheParametersIter<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

/// Information about an individual cache in the hierarchy.
///
/// # Platforms
/// 🟡 AMD ✅ Intel
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct CacheParameter {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

/// Info about a what a given cache caches (instructions, data, etc.)
#[derive(PartialEq, Eq, Debug)]
pub enum CacheType {
    /// Null - No more caches
    Null = 0,
    /// Data cache
    Data,
    /// Instruction cache
    Instruction,
    /// Data and Instruction cache
    Unified,
    /// 4-31 = Reserved
    Reserved,
}

impl fmt::Display for CacheType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let typ = match self {
            CacheType::Null => "Null",
            CacheType::Data => "Data",
            CacheType::Instruction => "Instruction",
            CacheType::Unified => "Unified",
            CacheType::Reserved => "Reserved",
        };

        f.write_str(typ)
    }
}

impl CacheParameter {
    /// Cache Type
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn cache_type(&self) -> CacheType {
        let typ = get_bits(self.eax, 0, 4) as u8;
        match typ {
            0 => CacheType::Null,
            1 => CacheType::Data,
            2 => CacheType::Instruction,
            3 => CacheType::Unified,
            _ => CacheType::Reserved,
        }
    }

    /// Cache Level (starts at 1)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn level(&self) -> u8 {
        get_bits(self.eax, 5, 7) as u8
    }

    /// Self Initializing cache level (does not need SW initialization).
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn is_self_initializing(&self) -> bool {
        get_bits(self.eax, 8, 8) == 1
    }

    /// Fully Associative cache
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn is_fully_associative(&self) -> bool {
        get_bits(self.eax, 9, 9) == 1
    }

    /// Maximum number of addressable IDs for logical processors sharing this cache
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn max_cores_for_cache(&self) -> usize {
        (get_bits(self.eax, 14, 25) + 1) as usize
    }

    /// Maximum number of addressable IDs for processor cores in the physical package
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn max_cores_for_package(&self) -> usize {
        (get_bits(self.eax, 26, 31) + 1) as usize
    }

    /// System Coherency Line Size (Bits 11-00)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn coherency_line_size(&self) -> usize {
        (get_bits(self.ebx, 0, 11) + 1) as usize
    }

    /// Physical Line partitions (Bits 21-12)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn physical_line_partitions(&self) -> usize {
        (get_bits(self.ebx, 12, 21) + 1) as usize
    }

    /// Ways of associativity (Bits 31-22)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn associativity(&self) -> usize {
        (get_bits(self.ebx, 22, 31) + 1) as usize
    }

    /// Number of Sets (Bits 31-00)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn sets(&self) -> usize {
        (self.ecx + 1) as usize
    }

    /// Write-Back Invalidate/Invalidate (Bit 0)
    /// False: WBINVD/INVD from threads sharing this cache acts upon lower level caches for threads sharing this cache.
    /// True: WBINVD/INVD is not guaranteed to act upon lower level caches of non-originating threads sharing this cache.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn is_write_back_invalidate(&self) -> bool {
        get_bits(self.edx, 0, 0) == 1
    }

    /// Cache Inclusiveness (Bit 1)
    /// False: Cache is not inclusive of lower cache levels.
    /// True: Cache is inclusive of lower cache levels.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn is_inclusive(&self) -> bool {
        get_bits(self.edx, 1, 1) == 1
    }

    /// Complex Cache Indexing (Bit 2)
    /// False: Direct mapped cache.
    /// True: A complex function is used to index the cache, potentially using all address bits.
    ///
    /// # Platforms
    /// ❌ AMD ✅ Intel
    pub fn has_complex_indexing(&self) -> bool {
        get_bits(self.edx, 2, 2) == 1
    }
}

impl Debug for CacheParameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CacheParameter")
            .field("cache_type", &self.cache_type())
            .field("level", &self.level())
            .field("is_self_initializing", &self.is_self_initializing())
            .field("is_fully_associative", &self.is_fully_associative())
            .field("max_cores_for_cache", &self.max_cores_for_cache())
            .field("max_cores_for_package", &self.max_cores_for_package())
            .field("coherency_line_size", &self.coherency_line_size())
            .field("physical_line_partitions", &self.physical_line_partitions())
            .field("associativity", &self.associativity())
            .field("sets", &self.sets())
            .field("is_write_back_invalidate", &self.is_write_back_invalidate())
            .field("is_inclusive", &self.is_inclusive())
            .field("has_complex_indexing", &self.has_complex_indexing())
            .finish()
    }
}

/// Information about how monitor/mwait works on this CPU (LEAF=0x05).
///
/// # Platforms
/// 🟡 AMD ✅ Intel
#[derive(Eq, PartialEq)]
pub struct MonitorMwaitInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl MonitorMwaitInfo {
    /// Smallest monitor-line size in bytes (default is processor's monitor granularity)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn smallest_monitor_line(&self) -> u16 {
        get_bits(self.eax, 0, 15) as u16
    }

    /// Largest monitor-line size in bytes (default is processor's monitor granularity)
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn largest_monitor_line(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    ///  Enumeration of Monitor-Mwait extensions (beyond EAX and EBX registers) supported
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn extensions_supported(&self) -> bool {
        get_bits(self.ecx, 0, 0) == 1
    }

    ///  Supports treating interrupts as break-event for MWAIT, even when interrupts disabled
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn interrupts_as_break_event(&self) -> bool {
        get_bits(self.ecx, 1, 1) == 1
    }

    /// Number of C0 sub C-states supported using MWAIT (Bits 03 - 00)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c0_states(&self) -> u16 {
        get_bits(self.edx, 0, 3) as u16
    }

    /// Number of C1 sub C-states supported using MWAIT (Bits 07 - 04)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c1_states(&self) -> u16 {
        get_bits(self.edx, 4, 7) as u16
    }

    /// Number of C2 sub C-states supported using MWAIT (Bits 11 - 08)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c2_states(&self) -> u16 {
        get_bits(self.edx, 8, 11) as u16
    }

    /// Number of C3 sub C-states supported using MWAIT (Bits 15 - 12)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c3_states(&self) -> u16 {
        get_bits(self.edx, 12, 15) as u16
    }

    /// Number of C4 sub C-states supported using MWAIT (Bits 19 - 16)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c4_states(&self) -> u16 {
        get_bits(self.edx, 16, 19) as u16
    }

    /// Number of C5 sub C-states supported using MWAIT (Bits 23 - 20)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c5_states(&self) -> u16 {
        get_bits(self.edx, 20, 23) as u16
    }

    /// Number of C6 sub C-states supported using MWAIT (Bits 27 - 24)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c6_states(&self) -> u16 {
        get_bits(self.edx, 24, 27) as u16
    }

    /// Number of C7 sub C-states supported using MWAIT (Bits 31 - 28)
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn supported_c7_states(&self) -> u16 {
        get_bits(self.edx, 28, 31) as u16
    }
}

impl Debug for MonitorMwaitInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MonitorMwaitInfo")
            .field("smallest_monitor_line", &self.smallest_monitor_line())
            .field("largest_monitor_line", &self.largest_monitor_line())
            .field("extensions_supported", &self.extensions_supported())
            .field(
                "interrupts_as_break_event",
                &self.interrupts_as_break_event(),
            )
            .field("supported_c0_states", &self.supported_c0_states())
            .field("supported_c1_states", &self.supported_c1_states())
            .field("supported_c2_states", &self.supported_c2_states())
            .field("supported_c3_states", &self.supported_c3_states())
            .field("supported_c4_states", &self.supported_c4_states())
            .field("supported_c5_states", &self.supported_c5_states())
            .field("supported_c6_states", &self.supported_c6_states())
            .field("supported_c7_states", &self.supported_c7_states())
            .finish()
    }
}

/// Query information about thermal and power management features of the CPU (LEAF=0x06).
///
/// # Platforms
/// 🟡 AMD ✅ Intel
pub struct ThermalPowerInfo {
    eax: ThermalPowerFeaturesEax,
    ebx: u32,
    ecx: ThermalPowerFeaturesEcx,
    _edx: u32,
}

impl ThermalPowerInfo {
    /// Number of Interrupt Thresholds in Digital Thermal Sensor
    ///
    /// # Platforms
    /// ❌ AMD (undefined/reserved) ✅ Intel
    pub fn dts_irq_threshold(&self) -> u8 {
        get_bits(self.ebx, 0, 3) as u8
    }

    /// Digital temperature sensor is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_dts(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::DTS)
    }

    /// Intel Turbo Boost Technology Available (see description of
    /// IA32_MISC_ENABLE\[38\]).
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_turbo_boost(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::TURBO_BOOST)
    }

    /// ARAT. APIC-Timer-always-running feature is supported if set.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_arat(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::ARAT)
    }

    /// PLN. Power limit notification controls are supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_pln(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::PLN)
    }

    /// ECMD. Clock modulation duty cycle extension is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_ecmd(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::ECMD)
    }

    /// PTM. Package thermal management is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_ptm(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::PTM)
    }

    /// HWP. HWP base registers (IA32_PM_ENABLE[bit 0], IA32_HWP_CAPABILITIES,
    /// IA32_HWP_REQUEST, IA32_HWP_STATUS) are supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::HWP)
    }

    /// HWP Notification. IA32_HWP_INTERRUPT MSR is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_notification(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::HWP_NOTIFICATION)
    }

    /// HWP Activity Window. IA32_HWP_REQUEST[bits 41:32] is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_activity_window(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::HWP_ACTIVITY_WINDOW)
    }

    /// HWP Energy Performance Preference. IA32_HWP_REQUEST[bits 31:24] is
    /// supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_energy_performance_preference(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::HWP_ENERGY_PERFORMANCE_PREFERENCE)
    }

    /// HWP Package Level Request. IA32_HWP_REQUEST_PKG MSR is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_package_level_request(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::HWP_PACKAGE_LEVEL_REQUEST)
    }

    /// HDC. HDC base registers IA32_PKG_HDC_CTL, IA32_PM_CTL1,
    /// IA32_THREAD_STALL MSRs are supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hdc(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::HDC)
    }

    /// Intel® Turbo Boost Max Technology 3.0 available.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_turbo_boost3(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::TURBO_BOOST_3)
    }

    /// HWP Capabilities. Highest Performance change is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_capabilities(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::HWP_CAPABILITIES)
    }

    /// HWP PECI override is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_peci_override(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::HWP_PECI_OVERRIDE)
    }

    /// Flexible HWP is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_flexible_hwp(&self) -> bool {
        self.eax.contains(ThermalPowerFeaturesEax::FLEXIBLE_HWP)
    }

    /// Fast access mode for the IA32_HWP_REQUEST MSR is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_hwp_fast_access_mode(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::HWP_REQUEST_MSR_FAST_ACCESS)
    }

    /// Ignoring Idle Logical Processor HWP request is supported if set.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_ignore_idle_processor_hwp_request(&self) -> bool {
        self.eax
            .contains(ThermalPowerFeaturesEax::IGNORE_IDLE_PROCESSOR_HWP_REQUEST)
    }

    /// Hardware Coordination Feedback Capability
    ///
    /// Presence of IA32_MPERF and IA32_APERF.
    ///
    /// The capability to provide a measure of delivered processor performance
    /// (since last reset of the counters), as a percentage of expected
    /// processor performance at frequency specified in CPUID Brand String Bits
    /// 02 - 01
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    pub fn has_hw_coord_feedback(&self) -> bool {
        self.ecx
            .contains(ThermalPowerFeaturesEcx::HW_COORD_FEEDBACK)
    }

    /// The processor supports performance-energy bias preference if
    /// CPUID.06H:ECX.SETBH[bit 3] is set and it also implies the presence of a
    /// new architectural MSR called IA32_ENERGY_PERF_BIAS (1B0H)
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    pub fn has_energy_bias_pref(&self) -> bool {
        self.ecx.contains(ThermalPowerFeaturesEcx::ENERGY_BIAS_PREF)
    }
}

impl Debug for ThermalPowerInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ThermalPowerInfo")
            .field("dts_irq_threshold", &self.dts_irq_threshold())
            .field("has_dts", &self.has_dts())
            .field("has_arat", &self.has_arat())
            .field("has_pln", &self.has_pln())
            .field("has_ecmd", &self.has_ecmd())
            .field("has_ptm", &self.has_ptm())
            .field("has_hwp", &self.has_hwp())
            .field("has_hwp_notification", &self.has_hwp_notification())
            .field("has_hwp_activity_window", &self.has_hwp_activity_window())
            .field(
                "has_hwp_energy_performance_preference",
                &self.has_hwp_energy_performance_preference(),
            )
            .field(
                "has_hwp_package_level_request",
                &self.has_hwp_package_level_request(),
            )
            .field("has_hdc", &self.has_hdc())
            .field("has_turbo_boost3", &self.has_turbo_boost3())
            .field("has_hwp_capabilities", &self.has_hwp_capabilities())
            .field("has_hwp_peci_override", &self.has_hwp_peci_override())
            .field("has_flexible_hwp", &self.has_flexible_hwp())
            .field("has_hwp_fast_access_mode", &self.has_hwp_fast_access_mode())
            .field(
                "has_ignore_idle_processor_hwp_request",
                &self.has_ignore_idle_processor_hwp_request(),
            )
            .field("has_hw_coord_feedback", &self.has_hw_coord_feedback())
            .field("has_energy_bias_pref", &self.has_energy_bias_pref())
            .finish()
    }
}

bitflags! {
    struct ThermalPowerFeaturesEax: u32 {
        /// Digital temperature sensor is supported if set. (Bit 00)
        const DTS = 1 << 0;
        /// Intel Turbo Boost Technology Available (see description of IA32_MISC_ENABLE[38]). (Bit 01)
        const TURBO_BOOST = 1 << 1;
        /// ARAT. APIC-Timer-always-running feature is supported if set. (Bit 02)
        const ARAT = 1 << 2;
        /// Bit 3: Reserved.
        const RESERVED_3 = 1 << 3;
        /// PLN. Power limit notification controls are supported if set. (Bit 04)
        const PLN = 1 << 4;
        /// ECMD. Clock modulation duty cycle extension is supported if set. (Bit 05)
        const ECMD = 1 << 5;
        /// PTM. Package thermal management is supported if set. (Bit 06)
        const PTM = 1 << 6;
        /// Bit 07: HWP. HWP base registers (IA32_PM_ENABLE[bit 0], IA32_HWP_CAPABILITIES, IA32_HWP_REQUEST, IA32_HWP_STATUS) are supported if set.
        const HWP = 1 << 7;
        /// Bit 08: HWP_Notification. IA32_HWP_INTERRUPT MSR is supported if set.
        const HWP_NOTIFICATION = 1 << 8;
        /// Bit 09: HWP_Activity_Window. IA32_HWP_REQUEST[bits 41:32] is supported if set.
        const HWP_ACTIVITY_WINDOW = 1 << 9;
        /// Bit 10: HWP_Energy_Performance_Preference. IA32_HWP_REQUEST[bits 31:24] is supported if set.
        const HWP_ENERGY_PERFORMANCE_PREFERENCE = 1 << 10;
        /// Bit 11: HWP_Package_Level_Request. IA32_HWP_REQUEST_PKG MSR is supported if set.
        const HWP_PACKAGE_LEVEL_REQUEST = 1 << 11;
        /// Bit 12: Reserved.
        const RESERVED_12 = 1 << 12;
        /// Bit 13: HDC. HDC base registers IA32_PKG_HDC_CTL, IA32_PM_CTL1, IA32_THREAD_STALL MSRs are supported if set.
        const HDC = 1 << 13;
        /// Bit 14: Intel® Turbo Boost Max Technology 3.0 available.
        const TURBO_BOOST_3 = 1 << 14;
        /// Bit 15: HWP Capabilities. Highest Performance change is supported if set.
        const HWP_CAPABILITIES = 1 << 15;
        /// Bit 16: HWP PECI override is supported if set.
        const HWP_PECI_OVERRIDE = 1 << 16;
        /// Bit 17: Flexible HWP is supported if set.
        const FLEXIBLE_HWP = 1 << 17;
        /// Bit 18: Fast access mode for the IA32_HWP_REQUEST MSR is supported if set.
        const HWP_REQUEST_MSR_FAST_ACCESS = 1 << 18;
        /// Bit 19: Reserved.
        const RESERVED_19 = 1 << 19;
        /// Bit 20: Ignoring Idle Logical Processor HWP request is supported if set.
        const IGNORE_IDLE_PROCESSOR_HWP_REQUEST = 1 << 20;
        // Bits 31 - 21: Reserved
    }
}

bitflags! {
    struct ThermalPowerFeaturesEcx: u32 {
        const HW_COORD_FEEDBACK = 1 << 0;

        /// The processor supports performance-energy bias preference if CPUID.06H:ECX.SETBH[bit 3] is set and it also implies the presence of a new architectural MSR called IA32_ENERGY_PERF_BIAS (1B0H)
        const ENERGY_BIAS_PREF = 1 << 3;
    }
}

/// Structured Extended Feature Identifiers (LEAF=0x07).
///
/// # Platforms
/// 🟡 AMD ✅ Intel
pub struct ExtendedFeatures {
    _eax: u32,
    ebx: ExtendedFeaturesEbx,
    ecx: ExtendedFeaturesEcx,
    edx: ExtendedFeaturesEdx,
    eax1: ExtendedFeaturesEax1,
    _ebx1: u32,
    _ecx1: u32,
    edx1: ExtendedFeaturesEdx1,
}

impl ExtendedFeatures {
    /// FSGSBASE. Supports RDFSBASE/RDGSBASE/WRFSBASE/WRGSBASE if 1.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_fsgsbase(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::FSGSBASE)
    }

    /// IA32_TSC_ADJUST MSR is supported if 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_tsc_adjust_msr(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::ADJUST_MSR)
    }

    /// BMI1
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_bmi1(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::BMI1)
    }

    /// HLE
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_hle(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::HLE)
    }

    /// AVX2
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx2(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX2)
    }

    /// FDP_EXCPTN_ONLY. x87 FPU Data Pointer updated only on x87 exceptions if
    /// 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_fdp(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::FDP)
    }

    /// SMEP. Supports Supervisor-Mode Execution Prevention if 1.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_smep(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::SMEP)
    }

    /// BMI2
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_bmi2(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::BMI2)
    }

    /// Supports Enhanced REP MOVSB/STOSB if 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_rep_movsb_stosb(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::REP_MOVSB_STOSB)
    }

    /// INVPCID. If 1, supports INVPCID instruction for system software that
    /// manages process-context identifiers.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_invpcid(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::INVPCID)
    }

    /// RTM
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_rtm(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::RTM)
    }

    /// Supports Intel Resource Director Technology (RDT) Monitoring capability.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_rdtm(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::RDTM)
    }

    /// Deprecates FPU CS and FPU DS values if 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_fpu_cs_ds_deprecated(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::DEPRECATE_FPU_CS_DS)
    }

    /// MPX. Supports Intel Memory Protection Extensions if 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_mpx(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::MPX)
    }

    /// Supports Intel Resource Director Technology (RDT) Allocation capability.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_rdta(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::RDTA)
    }

    /// Supports RDSEED.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_rdseed(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::RDSEED)
    }

    /// Supports ADX.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_adx(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::ADX)
    }

    /// SMAP. Supports Supervisor-Mode Access Prevention (and the CLAC/STAC
    /// instructions) if 1.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_smap(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::SMAP)
    }

    /// Supports CLFLUSHOPT.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_clflushopt(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::CLFLUSHOPT)
    }

    /// Supports Intel Processor Trace.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_processor_trace(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::PROCESSOR_TRACE)
    }

    /// Supports SHA Instructions.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_sha(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::SHA)
    }

    /// Supports Intel® Software Guard Extensions (Intel® SGX Extensions).
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_sgx(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::SGX)
    }

    /// Supports AVX512F.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512f(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512F)
    }

    /// Supports AVX512DQ.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512dq(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512DQ)
    }

    /// AVX512_IFMA
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512_ifma(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512_IFMA)
    }

    /// AVX512PF
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512pf(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512PF)
    }

    /// AVX512ER
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512er(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512ER)
    }

    /// AVX512CD
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512cd(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512CD)
    }

    /// AVX512BW
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512bw(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512BW)
    }

    /// AVX512VL
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512vl(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::AVX512VL)
    }

    /// CLWB
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_clwb(&self) -> bool {
        self.ebx.contains(ExtendedFeaturesEbx::CLWB)
    }

    /// Has PREFETCHWT1 (Intel® Xeon Phi™ only).
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_prefetchwt1(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::PREFETCHWT1)
    }

    /// AVX512VBMI
    ///
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512vbmi(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512VBMI)
    }

    /// Supports user-mode instruction prevention if 1.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_umip(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::UMIP)
    }

    /// Supports protection keys for user-mode pages.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_pku(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::PKU)
    }

    /// OS has set CR4.PKE to enable protection keys (and the RDPKRU/WRPKRU
    /// instructions.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_ospke(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::OSPKE)
    }

    /// WAITPKG
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_waitpkg(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::WAITPKG)
    }

    /// AVX512VBMI2
    ///
    /// ✅ AMD ✅ Intel
    #[deprecated(since = "11.4.0", note = "Please use `has_avx512vbmi2` instead")]
    #[inline]
    pub const fn has_av512vbmi2(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512VBMI2)
    }

    /// AVX512VBMI2
    ///
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512vbmi2(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512VBMI2)
    }

    /// Supports CET shadow stack features. Processors that set this bit define bits 0..2 of the
    /// IA32_U_CET and IA32_S_CET MSRs. Enumerates support for the following MSRs:
    /// IA32_INTERRUPT_SPP_TABLE_ADDR, IA32_PL3_SSP, IA32_PL2_SSP, IA32_PL1_SSP, and IA32_PL0_SSP.
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_cet_ss(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::CETSS)
    }

    /// GFNI
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_gfni(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::GFNI)
    }

    /// VAES
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_vaes(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::VAES)
    }

    /// VPCLMULQDQ
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_vpclmulqdq(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::VPCLMULQDQ)
    }

    /// AVX512VNNI
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512vnni(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512VNNI)
    }

    /// AVX512BITALG
    ///
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512bitalg(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512BITALG)
    }

    /// Indicates the following MSRs are supported: IA32_TME_CAPABILITY, IA32_TME_ACTIVATE,
    /// IA32_TME_EXCLUDE_MASK, and IA32_TME_EXCLUDE_BASE.
    ///
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_tme_en(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::TMEEN)
    }

    /// AVX512VPOPCNTDQ
    ///
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_avx512vpopcntdq(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::AVX512VPOPCNTDQ)
    }

    /// Supports 57-bit linear addresses and five-level paging if 1.
    ///
    /// # Platforms
    /// ❓ AMD ✅ Intel
    #[inline]
    pub const fn has_la57(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::LA57)
    }

    /// RDPID and IA32_TSC_AUX are available.
    ///
    /// # Bug
    /// The Intel manual lists RDPID as bit 22 in the ECX register, but AMD
    /// lists it as bit 22 in the ebx register. We assumed that the AMD manual
    /// was wrong and query ecx, let's see what happens.
    ///
    /// # Platforms
    /// ✅ AMD ✅ Intel
    #[inline]
    pub const fn has_rdpid(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::RDPID)
    }

    /// Supports SGX Launch Configuration.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_sgx_lc(&self) -> bool {
        self.ecx.contains(ExtendedFeaturesEcx::SGX_LC)
    }

    /// The value of MAWAU used by the BNDLDX and BNDSTX instructions in 64-bit mode.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub fn mawau_value(&self) -> u8 {
        get_bits(self.ecx.bits(), 17, 21) as u8
    }

    /// Supports AVX512_4VNNIW.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx512_4vnniw(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AVX512_4VNNIW)
    }

    /// Supports AVX512_4FMAPS.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx512_4fmaps(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AVX512_4FMAPS)
    }

    /// Supports AVX512_VP2INTERSECT.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx512_vp2intersect(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AVX512_VP2INTERSECT)
    }

    /// Supports AMX_BF16.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_amx_bf16(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AMX_BF16)
    }

    /// Supports AVX512_FP16.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx512_fp16(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AVX512_FP16)
    }

    /// Supports AMX_TILE.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_amx_tile(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AMX_TILE)
    }

    /// Supports AMX_INT8.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_amx_int8(&self) -> bool {
        self.edx.contains(ExtendedFeaturesEdx::AMX_INT8)
    }

    /// Supports AVX_VNNI.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx_vnni(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::AVX_VNNI)
    }

    /// Supports AVX512_BF16.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx512_bf16(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::AVX512_BF16)
    }

    /// Supports Fast zero-length REP MOVSB
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_fzrm(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::FZRM)
    }

    /// Supports Fast Short REP STOSB
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_fsrs(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::FSRS)
    }

    /// Supports Fast Short REP CMPSB, REP SCASB
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_fsrcrs(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::FSRCRS)
    }

    /// Supports HRESET
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_hreset(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::HRESET)
    }

    /// Supports AVX-IFMA Instructions.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx_ifma(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::AVX_IFMA)
    }

    /// Supports Linear Address Masking.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_lam(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::LAM)
    }

    /// Supports RDMSRLIST and WRMSRLIST Instructions and the IA32_BARRIER MSR.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_msrlist(&self) -> bool {
        self.eax1.contains(ExtendedFeaturesEax1::MSRLIST)
    }

    /// Supports INVD execution prevention after BIOS Done.
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_invd_disable_post_bios_done(&self) -> bool {
        self.eax1
            .contains(ExtendedFeaturesEax1::INVD_DISABLE_POST_BIOS_DONE)
    }

    /// Supports AVX_VNNI_INT8
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx_vnni_int8(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::AVX_VNNI_INT8)
    }

    /// Supports AVX_NE_CONVERT
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx_ne_convert(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::AVX_NE_CONVERT)
    }

    /// Supports AVX_VNNI_INT16
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx_vnni_int16(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::AVX_VNNI_INT16)
    }

    /// Supports PREFETCHI
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_prefetchi(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::PREFETCHI)
    }

    /// Supports UIRET_UIF
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_uiret_uif(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::UIRET_UIF)
    }

    /// Supports CET_SSS
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_cet_sss(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::CET_SSS)
    }

    /// Supports AVX10
    ///
    /// # Platforms
    /// ❌ AMD (reserved) ✅ Intel
    #[inline]
    pub const fn has_avx10(&self) -> bool {
        self.edx1.contains(ExtendedFeaturesEdx1::AVX10)
    }
}

impl Debug for ExtendedFeatures {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExtendedFeatures")
            .field("ebx", &self.ebx)
            .field("ecx", &self.ecx)
            .field("mawau_value", &self.mawau_value())
            .finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeaturesEbx: u32 {
        /// FSGSBASE. Supports RDFSBASE/RDGSBASE/WRFSBASE/WRGSBASE if 1. (Bit 00)
        const FSGSBASE = 1 << 0;
        /// IA32_TSC_ADJUST MSR is supported if 1. (Bit 01)
        const ADJUST_MSR = 1 << 1;
        /// Bit 02: SGX. Supports Intel® Software Guard Extensions (Intel® SGX Extensions) if 1.
        const SGX = 1 << 2;
        /// BMI1 (Bit 03)
        const BMI1 = 1 << 3;
        /// HLE (Bit 04)
        const HLE = 1 << 4;
        /// AVX2 (Bit 05)
        const AVX2 = 1 << 5;
        /// FDP_EXCPTN_ONLY. x87 FPU Data Pointer updated only on x87 exceptions if 1.
        const FDP = 1 << 6;
        /// SMEP. Supports Supervisor-Mode Execution Prevention if 1. (Bit 07)
        const SMEP = 1 << 7;
        /// BMI2 (Bit 08)
        const BMI2 = 1 << 8;
        /// Supports Enhanced REP MOVSB/STOSB if 1. (Bit 09)
        const REP_MOVSB_STOSB = 1 << 9;
        /// INVPCID. If 1, supports INVPCID instruction for system software that manages process-context identifiers. (Bit 10)
        const INVPCID = 1 << 10;
        /// RTM (Bit 11)
        const RTM = 1 << 11;
        /// Supports Intel Resource Director Technology (RDT) Monitoring. (Bit 12)
        const RDTM = 1 << 12;
        /// Deprecates FPU CS and FPU DS values if 1. (Bit 13)
        const DEPRECATE_FPU_CS_DS = 1 << 13;
        /// Supports Intel Memory Protection Extensions if set. (Bit 14)
        const MPX = 1 << 14;
        /// Supports Intel Resource Director Technology (RDT) Allocation capability if 1.
        const RDTA = 1 << 15;
        /// Bit 16: AVX512F.
        const AVX512F = 1 << 16;
        /// Bit 17: AVX512DQ.
        const AVX512DQ = 1 << 17;
        /// Supports RDSEED.
        const RDSEED = 1 << 18;
        /// Supports ADX.
        const ADX = 1 << 19;
        /// SMAP. Supports Supervisor-Mode Access Prevention (and the CLAC/STAC instructions) if 1.
        const SMAP = 1 << 20;
        /// Bit 21: AVX512_IFMA.
        const AVX512_IFMA = 1 << 21;
        // Bit 22: Reserved.
        /// Bit 23: CLFLUSHOPT
        const CLFLUSHOPT = 1 << 23;
        /// Bit 24: CLWB.
        const CLWB = 1 << 24;
        /// Bit 25: Intel Processor Trace
        const PROCESSOR_TRACE = 1 << 25;
        /// Bit 26: AVX512PF. (Intel® Xeon Phi™ only.)
        const AVX512PF = 1 << 26;
        /// Bit 27: AVX512ER. (Intel® Xeon Phi™ only.)
        const AVX512ER = 1 << 27;
        /// Bit 28: AVX512CD.
        const AVX512CD = 1 << 28;
        /// Bit 29: Intel SHA Extensions
        const SHA = 1 << 29;
        /// Bit 30: AVX512BW.
        const AVX512BW = 1 << 30;
        /// Bit 31: AVX512VL.
        const AVX512VL = 1 << 31;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeaturesEcx: u32 {
        /// Bit 0: Prefetch WT1. (Intel® Xeon Phi™ only).
        const PREFETCHWT1 = 1 << 0;
        // Bit 01: AVX512_VBMI
        const AVX512VBMI = 1 << 1;
        /// Bit 02: UMIP. Supports user-mode instruction prevention if 1.
        const UMIP = 1 << 2;
        /// Bit 03: PKU. Supports protection keys for user-mode pages if 1.
        const PKU = 1 << 3;
        /// Bit 04: OSPKE. If 1, OS has set CR4.PKE to enable protection keys (and the RDPKRU/WRPKRU instruc-tions).
        const OSPKE = 1 << 4;
        /// Bit 5: WAITPKG
        const WAITPKG = 1 << 5;
        /// Bit 6: AV512_VBMI2
        const AVX512VBMI2 = 1 << 6;
        /// Bit 7: CET_SS. Supports CET shadow stack features if 1. Processors that set this bit define bits 0..2 of the
        /// IA32_U_CET and IA32_S_CET MSRs. Enumerates support for the following MSRs:
        /// IA32_INTERRUPT_SPP_TABLE_ADDR, IA32_PL3_SSP, IA32_PL2_SSP, IA32_PL1_SSP, and IA32_PL0_SSP.
        const CETSS = 1 << 7;
        /// Bit 8: GFNI
        const GFNI = 1 << 8;
        /// Bit 9: VAES
        const VAES = 1 << 9;
        /// Bit 10: VPCLMULQDQ
        const VPCLMULQDQ = 1 << 10;
        /// Bit 11: AVX512_VNNI
        const AVX512VNNI = 1 << 11;
        /// Bit 12: AVX512_BITALG
        const AVX512BITALG = 1 << 12;
        /// Bit 13: TME_EN. If 1, the following MSRs are supported: IA32_TME_CAPABILITY, IA32_TME_ACTIVATE,
        /// IA32_TME_EXCLUDE_MASK, and IA32_TME_EXCLUDE_BASE.
        const TMEEN = 1 << 13;
        /// Bit 14: AVX512_VPOPCNTDQ
        const AVX512VPOPCNTDQ = 1 << 14;

        // Bit 15: Reserved.

        /// Bit 16: Supports 57-bit linear addresses and five-level paging if 1.
        const LA57 = 1 << 16;

        // Bits 21 - 17: The value of MAWAU used by the BNDLDX and BNDSTX instructions in 64-bit mode

        /// Bit 22: RDPID. RDPID and IA32_TSC_AUX are available if 1.
        const RDPID = 1 << 22;

        // Bits 29 - 23: Reserved.

        /// Bit 30: SGX_LC. Supports SGX Launch Configuration if 1.
        const SGX_LC = 1 << 30;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeaturesEdx: u32 {
        /// Bit 02: AVX512_4VNNIW. (Intel® Xeon Phi™ only).
        const AVX512_4VNNIW = 1 << 2;
        /// Bit 03: AVX512_4FMAPS. (Intel® Xeon Phi™ only).
        const AVX512_4FMAPS = 1 << 3;
        /// Bit 08: AVX512_VP2INTERSECT.
        const AVX512_VP2INTERSECT = 1 << 8;
        /// Bit 22: AMX-BF16. If 1, the processor supports tile computational operations on bfloat16 numbers.
        const AMX_BF16 = 1 << 22;
        /// Bit 23: AVX512_FP16.
        const AVX512_FP16 = 1 << 23;
        /// Bit 24: AMX-TILE. If 1, the processor supports tile architecture
        const AMX_TILE = 1 << 24;
        /// Bit 25: AMX-INT8. If 1, the processor supports tile computational operations on 8-bit integers.
        const AMX_INT8 = 1 << 25;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeaturesEax1: u32 {
        // Some of the Unimplemented bits are reserved and maybe release in future CPUs, see Intel SDM for future features (Date of comment: 07.17.2024)
        /// Bit 04: AVX_VNNI. AVX (VEX-encoded) versions of the Vector Neural Network Instructions.
        const AVX_VNNI = 1 << 4;
        /// Bit 05: AVX512_BF16. Vector Neural Network Instructions supporting BFLOAT16 inputs and conversion instructions from IEEE single precision.
        const AVX512_BF16 = 1 << 5;
        /// Bit 10: If 1, supports fast zero-length REP MOVSB.
        const FZRM = 1 << 10;
        /// Bit 11: If 1, supports fast short REP STOSB.
        const FSRS = 1 << 11;
        /// Bit 12: If 1, supports fast short REP CMPSB, REP SCASB.
        const FSRCRS = 1 << 12;
        /// Bit 22: If 1, supports history reset via the HRESET instruction and the IA32_HRESET_ENABLE MSR. When set, indicates that the Processor History Reset Leaf (EAX = 20H) is valid.
        const HRESET = 1 << 22;
        /// Bit 23: If 1, supports the AVX-IFMA instructions.
        const AVX_IFMA = 1 << 23;
        /// Bit 26: If 1, supports Linear Address Masking.
        const LAM = 1 << 26;
        /// Bit 27: If 1, supports the RDMSRLIST and WRMSRLIST instructions and the IA32_BARRIER MSR.
        const MSRLIST = 1 << 27;
        /// Bit 30: If 1, supports INVD execution prevention after BIOS Done.
        const INVD_DISABLE_POST_BIOS_DONE = 1 << 30;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedFeaturesEdx1: u32 {
        // Some of the Unimplemented bits are reserved and maybe release in future CPUs, see Intel SDM for future features (Date of comment: 07.17.2024)
        /// Bit 4: If 1, supports the AVX-VNNI-INT8 instructions.
        const AVX_VNNI_INT8 = 1 << 4;
        /// Bit 5: If 1, supports the AVX-NE-CONVERT instructions.
        const AVX_NE_CONVERT = 1 << 5;
        /// Bit 10: If 1, supports the AVX-VNNI-INT16 instructions
        const AVX_VNNI_INT16 = 1 << 10;
        /// Bit 14: If 1, supports the PREFETCHIT0/1 instructions
        const PREFETCHI = 1 << 14;
        /// Bit 17: If 1, UIRET sets UIF to the value of bit 1 of the RFLAGS image loaded from the stack
        const UIRET_UIF = 1 << 17;
        /// Bit 18: CET_SSS. If 1, indicates that an operating system can enable supervisor shadow stacks as long as it ensures that a supervisor shadow stack cannot become prematurely busy due to page faults
        const CET_SSS = 1 << 18;
        /// Bit 19: If 1, supports the Intel® AVX10 instructions and indicates the presence of CPUID Leaf 24H,
        /// which enumerates version number and supported vector lengths
        const AVX10 = 1 << 19;
    }
}

/// Direct cache access info (LEAF=0x09).
///
/// # Platforms
/// ❌ AMD (reserved) ✅ Intel
pub struct DirectCacheAccessInfo {
    eax: u32,
}

impl DirectCacheAccessInfo {
    /// Value of bits \[31:0\] of IA32_PLATFORM_DCA_CAP MSR (address 1F8H)
    pub fn get_dca_cap_value(&self) -> u32 {
        self.eax
    }
}

impl Debug for DirectCacheAccessInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DirectCacheAccessInfo")
            .field("dca_cap_value", &self.get_dca_cap_value())
            .finish()
    }
}

/// Info about performance monitoring -- how many counters etc. (LEAF=0x0A)
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct PerformanceMonitoringInfo {
    eax: u32,
    ebx: PerformanceMonitoringFeaturesEbx,
    _ecx: u32,
    edx: u32,
}

impl PerformanceMonitoringInfo {
    /// Version ID of architectural performance monitoring. (Bits 07 - 00)
    pub fn version_id(&self) -> u8 {
        get_bits(self.eax, 0, 7) as u8
    }

    /// Number of general-purpose performance monitoring counter per logical processor. (Bits 15- 08)
    pub fn number_of_counters(&self) -> u8 {
        get_bits(self.eax, 8, 15) as u8
    }

    /// Bit width of general-purpose, performance monitoring counter. (Bits 23 - 16)
    pub fn counter_bit_width(&self) -> u8 {
        get_bits(self.eax, 16, 23) as u8
    }

    /// Length of EBX bit vector to enumerate architectural performance monitoring events. (Bits 31 - 24)
    pub fn ebx_length(&self) -> u8 {
        get_bits(self.eax, 24, 31) as u8
    }

    /// Number of fixed-function performance counters (if Version ID > 1). (Bits 04 - 00)
    pub fn fixed_function_counters(&self) -> u8 {
        get_bits(self.edx, 0, 4) as u8
    }

    /// Bit width of fixed-function performance counters (if Version ID > 1). (Bits 12- 05)
    pub fn fixed_function_counters_bit_width(&self) -> u8 {
        get_bits(self.edx, 5, 12) as u8
    }

    check_bit_fn!(
        doc = "AnyThread deprecation",
        has_any_thread_deprecation,
        edx,
        15
    );

    check_flag!(
        doc = "Core cycle event not available if 1.",
        is_core_cyc_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::CORE_CYC_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Instruction retired event not available if 1.",
        is_inst_ret_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::INST_RET_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Reference cycles event not available if 1.",
        is_ref_cycle_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::REF_CYC_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Last-level cache reference event not available if 1.",
        is_cache_ref_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::CACHE_REF_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Last-level cache misses event not available if 1.",
        is_ll_cache_miss_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::LL_CACHE_MISS_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Branch instruction retired event not available if 1.",
        is_branch_inst_ret_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::BRANCH_INST_RET_EV_UNAVAILABLE
    );

    check_flag!(
        doc = "Branch mispredict retired event not available if 1.",
        is_branch_midpred_ev_unavailable,
        ebx,
        PerformanceMonitoringFeaturesEbx::BRANCH_MISPRED_EV_UNAVAILABLE
    );
}

impl Debug for PerformanceMonitoringInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PerformanceMonitoringInfo")
            .field("version_id", &self.version_id())
            .field("number_of_counters", &self.number_of_counters())
            .field("counter_bit_width", &self.counter_bit_width())
            .field("ebx_length", &self.ebx_length())
            .field("fixed_function_counters", &self.fixed_function_counters())
            .field(
                "fixed_function_counters_bit_width",
                &self.fixed_function_counters_bit_width(),
            )
            .finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct PerformanceMonitoringFeaturesEbx: u32 {
        /// Core cycle event not available if 1. (Bit 0)
        const CORE_CYC_EV_UNAVAILABLE = 1 << 0;
        /// Instruction retired event not available if 1. (Bit 01)
        const INST_RET_EV_UNAVAILABLE = 1 << 1;
        /// Reference cycles event not available if 1. (Bit 02)
        const REF_CYC_EV_UNAVAILABLE = 1 << 2;
        /// Last-level cache reference event not available if 1. (Bit 03)
        const CACHE_REF_EV_UNAVAILABLE = 1 << 3;
        /// Last-level cache misses event not available if 1. (Bit 04)
        const LL_CACHE_MISS_EV_UNAVAILABLE = 1 << 4;
        /// Branch instruction retired event not available if 1. (Bit 05)
        const BRANCH_INST_RET_EV_UNAVAILABLE = 1 << 5;
        /// Branch mispredict retired event not available if 1. (Bit 06)
        const BRANCH_MISPRED_EV_UNAVAILABLE = 1 << 6;
    }
}

/// Information about topology (LEAF=0x0B).
///
/// Iterates over the system topology in order to retrieve more system
/// information at each level of the topology: how many cores and what kind of
/// cores
///
/// # Platforms
/// ✅ AMD ✅ Intel
#[derive(Clone)]
pub struct ExtendedTopologyIter<R: CpuIdReader> {
    read: R,
    level: u32,
    is_v2: bool,
}

/// Gives information about the current level in the topology.
///
/// How many cores, what type etc.
#[derive(PartialEq, Eq)]
pub struct ExtendedTopologyLevel {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl fmt::Debug for ExtendedTopologyLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExtendedTopologyLevel")
            .field("processors", &self.processors())
            .field("number", &self.level_number())
            .field("type", &self.level_type())
            .field("x2apic_id", &self.x2apic_id())
            .field("next_apic_id", &self.shift_right_for_next_apic_id())
            .finish()
    }
}

impl ExtendedTopologyLevel {
    /// Number of logical processors at this level type.
    /// The number reflects configuration as shipped.
    pub fn processors(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    /// Level number.
    pub fn level_number(&self) -> u8 {
        get_bits(self.ecx, 0, 7) as u8
    }

    // Level type.
    pub fn level_type(&self) -> TopologyType {
        match get_bits(self.ecx, 8, 15) {
            0 => TopologyType::Invalid,
            1 => TopologyType::SMT,
            2 => TopologyType::Core,
            3 => TopologyType::Module,
            4 => TopologyType::Tile,
            5 => TopologyType::Die,
            _ => unreachable!(),
        }
    }

    /// x2APIC ID the current logical processor. (Bits 31-00)
    pub fn x2apic_id(&self) -> u32 {
        self.edx
    }

    /// Number of bits to shift right on x2APIC ID to get a unique topology ID of the next level type. (Bits 04-00)
    /// All logical processors with the same next level ID share current level.
    pub fn shift_right_for_next_apic_id(&self) -> u32 {
        get_bits(self.eax, 0, 4)
    }
}

/// What type of core we have at this level in the topology (real CPU or hyper-threaded).
#[derive(PartialEq, Eq, Debug)]
pub enum TopologyType {
    Invalid = 0,
    /// Hyper-thread (Simultaneous multithreading)
    SMT = 1,
    Core = 2,
    Module = 3,
    Tile = 4,
    Die = 5,
}

impl fmt::Display for TopologyType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data = match self {
            TopologyType::Invalid => "Invalid",
            TopologyType::SMT => "SMT",
            TopologyType::Core => "Core",
            TopologyType::Module => "Module",
            TopologyType::Tile => "Tile",
            TopologyType::Die => "Die",
        };

        f.write_str(data)
    }
}

impl<R: CpuIdReader> Iterator for ExtendedTopologyIter<R> {
    type Item = ExtendedTopologyLevel;

    fn next(&mut self) -> Option<ExtendedTopologyLevel> {
        let res = if self.is_v2 {
            self.read.cpuid2(EAX_EXTENDED_TOPOLOGY_INFO_V2, self.level)
        } else {
            self.read.cpuid2(EAX_EXTENDED_TOPOLOGY_INFO, self.level)
        };
        self.level += 1;

        let et = ExtendedTopologyLevel {
            eax: res.eax,
            ebx: res.ebx,
            ecx: res.ecx,
            edx: res.edx,
        };

        match et.level_type() {
            TopologyType::Invalid => None,
            _ => Some(et),
        }
    }
}

impl<R: CpuIdReader> Debug for ExtendedTopologyIter<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedStateInfoXCR0Flags: u32 {
        /// legacy x87 (Bit 00).
        const LEGACY_X87 = 1 << 0;

        /// 128-bit SSE (Bit 01).
        const SSE128 = 1 << 1;

        /// 256-bit AVX (Bit 02).
        const AVX256 = 1 << 2;

        /// MPX BNDREGS (Bit 03).
        const MPX_BNDREGS = 1 << 3;

        /// MPX BNDCSR (Bit 04).
        const MPX_BNDCSR = 1 << 4;

        /// AVX512 OPMASK (Bit 05).
        const AVX512_OPMASK = 1 << 5;

        /// AVX ZMM Hi256 (Bit 06).
        const AVX512_ZMM_HI256 = 1 << 6;

        /// AVX 512 ZMM Hi16 (Bit 07).
        const AVX512_ZMM_HI16 = 1 << 7;

        /// PKRU state (Bit 09).
        const PKRU = 1 << 9;

        /// IA32_XSS HDC State (Bit 13).
        const IA32_XSS_HDC = 1 << 13;

        /// AMX TILECFG state (Bit 17)
        const AMX_TILECFG = 1 << 17;

        /// AMX TILEDATA state (Bit 17)
        const AMX_TILEDATA = 1 << 18;
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ExtendedStateInfoXSSFlags: u32 {
        /// IA32_XSS PT (Trace Packet) State (Bit 08).
        const PT = 1 << 8;

        /// IA32_XSS PASID state (Bit 10)
        const PASID = 1 << 10;

        /// IA32_XSS CET user state (Bit 11)
        const CET_USER = 1 << 11;

        /// IA32_XSS CET supervisor state (Bit 12)
        const CET_SUPERVISOR = 1 << 12;

        /// IA32_XSS HDC State (Bit 13).
        const HDC = 1 << 13;

        /// IA32_XSS UINTR state (Bit 14)
        const UINTR = 1 << 14;

        /// IA32_XSS LBR state (Bit 15)
        const LBR = 1 << 15;

        /// IA32_XSS HWP state (Bit 16)
        const HWP = 1 << 16;
    }
}

/// Information for saving/restoring extended register state (LEAF=0x0D).
///
/// # Platforms
/// ✅ AMD ✅ Intel
pub struct ExtendedStateInfo<R: CpuIdReader> {
    read: R,
    eax: ExtendedStateInfoXCR0Flags,
    ebx: u32,
    ecx: u32,
    _edx: u32,
    eax1: u32,
    ebx1: u32,
    ecx1: ExtendedStateInfoXSSFlags,
    _edx1: u32,
}

impl<F: CpuIdReader> ExtendedStateInfo<F> {
    check_flag!(
        doc = "Support for legacy x87 in XCR0.",
        xcr0_supports_legacy_x87,
        eax,
        ExtendedStateInfoXCR0Flags::LEGACY_X87
    );

    check_flag!(
        doc = "Support for SSE 128-bit in XCR0.",
        xcr0_supports_sse_128,
        eax,
        ExtendedStateInfoXCR0Flags::SSE128
    );

    check_flag!(
        doc = "Support for AVX 256-bit in XCR0.",
        xcr0_supports_avx_256,
        eax,
        ExtendedStateInfoXCR0Flags::AVX256
    );

    check_flag!(
        doc = "Support for MPX BNDREGS in XCR0.",
        xcr0_supports_mpx_bndregs,
        eax,
        ExtendedStateInfoXCR0Flags::MPX_BNDREGS
    );

    check_flag!(
        doc = "Support for MPX BNDCSR in XCR0.",
        xcr0_supports_mpx_bndcsr,
        eax,
        ExtendedStateInfoXCR0Flags::MPX_BNDCSR
    );

    check_flag!(
        doc = "Support for AVX512 OPMASK in XCR0.",
        xcr0_supports_avx512_opmask,
        eax,
        ExtendedStateInfoXCR0Flags::AVX512_OPMASK
    );

    check_flag!(
        doc = "Support for AVX512 ZMM Hi256 XCR0.",
        xcr0_supports_avx512_zmm_hi256,
        eax,
        ExtendedStateInfoXCR0Flags::AVX512_ZMM_HI256
    );

    check_flag!(
        doc = "Support for AVX512 ZMM Hi16 in XCR0.",
        xcr0_supports_avx512_zmm_hi16,
        eax,
        ExtendedStateInfoXCR0Flags::AVX512_ZMM_HI16
    );

    check_flag!(
        doc = "Support for PKRU in XCR0.",
        xcr0_supports_pkru,
        eax,
        ExtendedStateInfoXCR0Flags::PKRU
    );

    check_flag!(
        doc = "Support for PT in IA32_XSS.",
        ia32_xss_supports_pt,
        ecx1,
        ExtendedStateInfoXSSFlags::PT
    );

    check_flag!(
        doc = "Support for HDC in IA32_XSS.",
        ia32_xss_supports_hdc,
        ecx1,
        ExtendedStateInfoXSSFlags::HDC
    );

    /// Maximum size (bytes, from the beginning of the XSAVE/XRSTOR save area) required by
    /// enabled features in XCR0. May be different than ECX if some features at the end of the XSAVE save area
    /// are not enabled.
    pub fn xsave_area_size_enabled_features(&self) -> u32 {
        self.ebx
    }

    /// Maximum size (bytes, from the beginning of the XSAVE/XRSTOR save area) of the
    /// XSAVE/XRSTOR save area required by all supported features in the processor,
    /// i.e all the valid bit fields in XCR0.
    pub fn xsave_area_size_supported_features(&self) -> u32 {
        self.ecx
    }

    /// CPU has xsaveopt feature.
    pub fn has_xsaveopt(&self) -> bool {
        self.eax1 & 0x1 > 0
    }

    /// Supports XSAVEC and the compacted form of XRSTOR if set.
    pub fn has_xsavec(&self) -> bool {
        self.eax1 & 0b10 > 0
    }

    /// Supports XGETBV with ECX = 1 if set.
    pub fn has_xgetbv(&self) -> bool {
        self.eax1 & 0b100 > 0
    }

    /// Supports XSAVES/XRSTORS and IA32_XSS if set.
    pub fn has_xsaves_xrstors(&self) -> bool {
        self.eax1 & 0b1000 > 0
    }

    /// The size in bytes of the XSAVE area containing all states enabled by XCRO | IA32_XSS.
    pub fn xsave_size(&self) -> u32 {
        self.ebx1
    }

    /// Iterator over extended state enumeration levels >= 2.
    pub fn iter(&self) -> ExtendedStateIter<F> {
        ExtendedStateIter {
            read: self.read.clone(),
            level: 1,
            supported_xcr0: self.eax.bits(),
            supported_xss: self.ecx1.bits(),
        }
    }
}

impl<R: CpuIdReader> Debug for ExtendedStateInfo<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExtendedStateInfo")
            .field("eax", &self.eax)
            .field("ecx1", &self.ecx1)
            .field(
                "xsave_area_size_enabled_features",
                &self.xsave_area_size_enabled_features(),
            )
            .field(
                "xsave_area_size_supported_features",
                &self.xsave_area_size_supported_features(),
            )
            .field("has_xsaveopt", &self.has_xsaveopt())
            .field("has_xsavec", &self.has_xsavec())
            .field("has_xgetbv", &self.has_xgetbv())
            .field("has_xsaves_xrstors", &self.has_xsaves_xrstors())
            .field("xsave_size", &self.xsave_size())
            .field("extended_state_iter", &self.iter())
            .finish()
    }
}

/// Yields [ExtendedState] structs.
#[derive(Clone)]
pub struct ExtendedStateIter<R: CpuIdReader> {
    read: R,
    level: u32,
    supported_xcr0: u32,
    supported_xss: u32,
}

/// When CPUID executes with EAX set to 0DH and ECX = n (n > 1, and is a valid
/// sub-leaf index), the processor returns information about the size and offset
/// of each processor extended state save area within the XSAVE/XRSTOR area.
///
/// The iterator goes over the valid sub-leaves and obtain size and offset
/// information for each processor extended state save area:
impl<R: CpuIdReader> Iterator for ExtendedStateIter<R> {
    type Item = ExtendedState;

    fn next(&mut self) -> Option<ExtendedState> {
        self.level += 1;
        if self.level > 31 {
            return None;
        }

        let bit = 1 << self.level;
        if (self.supported_xcr0 & bit > 0) || (self.supported_xss & bit > 0) {
            let res = self.read.cpuid2(EAX_EXTENDED_STATE_INFO, self.level);
            return Some(ExtendedState {
                subleaf: self.level,
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
            });
        }

        self.next()
    }
}

impl<R: CpuIdReader> Debug for ExtendedStateIter<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

/// What kidn of extended register state this is.
#[derive(PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum ExtendedRegisterType {
    Avx,
    MpxBndregs,
    MpxBndcsr,
    Avx512Opmask,
    Avx512ZmmHi256,
    Avx512ZmmHi16,
    Pt,
    Pkru,
    Hdc,
    Unknown(u32),
}

impl From<u32> for ExtendedRegisterType {
    fn from(value: u32) -> ExtendedRegisterType {
        match value {
            0x2 => ExtendedRegisterType::Avx,
            0x3 => ExtendedRegisterType::MpxBndregs,
            0x4 => ExtendedRegisterType::MpxBndcsr,
            0x5 => ExtendedRegisterType::Avx512Opmask,
            0x6 => ExtendedRegisterType::Avx512ZmmHi256,
            0x7 => ExtendedRegisterType::Avx512ZmmHi16,
            0x8 => ExtendedRegisterType::Pt,
            0x9 => ExtendedRegisterType::Pkru,
            0xd => ExtendedRegisterType::Hdc,
            x => ExtendedRegisterType::Unknown(x),
        }
    }
}

impl fmt::Display for ExtendedRegisterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data = match self {
            ExtendedRegisterType::Avx => "AVX/YMM",
            ExtendedRegisterType::MpxBndregs => "MPX BNDREGS",
            ExtendedRegisterType::MpxBndcsr => "MPX BNDCSR",
            ExtendedRegisterType::Avx512Opmask => "AVX-512 opmask",
            ExtendedRegisterType::Avx512ZmmHi256 => "AVX-512 ZMM_Hi256",
            ExtendedRegisterType::Avx512ZmmHi16 => "AVX-512 Hi16_ZMM",
            ExtendedRegisterType::Pkru => "PKRU",
            ExtendedRegisterType::Pt => "PT",
            ExtendedRegisterType::Hdc => "HDC",
            ExtendedRegisterType::Unknown(t) => {
                return write!(f, "Unknown({})", t);
            }
        };

        f.write_str(data)
    }
}

/// Where the extended register state is stored.
#[derive(PartialEq, Eq, Debug)]
pub enum ExtendedRegisterStateLocation {
    Xcr0,
    Ia32Xss,
}

impl fmt::Display for ExtendedRegisterStateLocation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data = match self {
            ExtendedRegisterStateLocation::Xcr0 => "XCR0 (user state)",
            ExtendedRegisterStateLocation::Ia32Xss => "IA32_XSS (supervisor state)",
        };

        f.write_str(data)
    }
}

/// ExtendedState subleaf structure for things that need to be restored.
pub struct ExtendedState {
    pub subleaf: u32,
    eax: u32,
    ebx: u32,
    ecx: u32,
}

impl ExtendedState {
    /// Returns which register this specific extended subleaf contains information for.
    pub fn register(&self) -> ExtendedRegisterType {
        self.subleaf.into()
    }

    /// The size in bytes (from the offset specified in EBX) of the save area
    /// for an extended state feature associated with a valid sub-leaf index, n.
    /// This field reports 0 if the sub-leaf index, n, is invalid.
    pub fn size(&self) -> u32 {
        self.eax
    }

    /// The offset in bytes of this extended state components save area
    /// from the beginning of the XSAVE/XRSTOR area.
    pub fn offset(&self) -> u32 {
        self.ebx
    }

    pub fn location(&self) -> ExtendedRegisterStateLocation {
        if self.is_in_xcr0() {
            ExtendedRegisterStateLocation::Xcr0
        } else {
            ExtendedRegisterStateLocation::Ia32Xss
        }
    }

    /// True if the bit n (corresponding to the sub-leaf index)
    /// is supported in the IA32_XSS MSR;
    ///
    /// # Deprecation note
    /// This will likely be removed in the future. Use `location()` instead.
    pub fn is_in_ia32_xss(&self) -> bool {
        self.ecx & 0b1 > 0
    }

    /// True if bit n is supported in XCR0.
    ///
    /// # Deprecation note
    /// This will likely be removed in the future. Use `location()` instead.
    pub fn is_in_xcr0(&self) -> bool {
        self.ecx & 0b1 == 0
    }

    /// Returns true when the compacted format of an XSAVE area is used,
    /// this extended state component located on the next 64-byte
    /// boundary following the preceding state component
    /// (otherwise, it is located immediately following the preceding state component).
    pub fn is_compacted_format(&self) -> bool {
        self.ecx & 0b10 > 0
    }
}

impl Debug for ExtendedState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtendedState")
            .field("size", &self.size())
            .field("offset", &self.offset())
            .field("is_in_ia32_xss", &self.is_in_ia32_xss())
            .field("is_in_xcr0", &self.is_in_xcr0())
            .field("is_compacted_format", &self.is_compacted_format())
            .finish()
    }
}

/// Intel Resource Director Technology RDT (LEAF=0x0F).
///
/// Monitoring Enumeration Sub-leaf (EAX = 0FH, ECX = 0 and ECX = 1)
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct RdtMonitoringInfo<R: CpuIdReader> {
    read: R,
    ebx: u32,
    edx: u32,
}

impl<R: CpuIdReader> RdtMonitoringInfo<R> {
    /// Maximum range (zero-based) of RMID within this physical processor of all types.
    pub fn rmid_range(&self) -> u32 {
        self.ebx
    }

    check_bit_fn!(
        doc = "Supports L3 Cache Intel RDT Monitoring.",
        has_l3_monitoring,
        edx,
        1
    );

    /// L3 Cache Monitoring.
    pub fn l3_monitoring(&self) -> Option<L3MonitoringInfo> {
        if self.has_l3_monitoring() {
            let res = self.read.cpuid2(EAX_RDT_MONITORING, 1);
            Some(L3MonitoringInfo {
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }
}

impl<R: CpuIdReader> Debug for RdtMonitoringInfo<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdtMonitoringInfo")
            .field("rmid_range", &self.rmid_range())
            .field("l3_monitoring", &self.l3_monitoring())
            .finish()
    }
}

/// Information about L3 cache monitoring.
pub struct L3MonitoringInfo {
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl L3MonitoringInfo {
    /// Conversion factor from reported IA32_QM_CTR value to occupancy metric (bytes).
    pub fn conversion_factor(&self) -> u32 {
        self.ebx
    }

    /// Maximum range (zero-based) of RMID of L3.
    pub fn maximum_rmid_range(&self) -> u32 {
        self.ecx
    }

    check_bit_fn!(
        doc = "Supports occupancy monitoring.",
        has_occupancy_monitoring,
        edx,
        0
    );

    check_bit_fn!(
        doc = "Supports total bandwidth monitoring.",
        has_total_bandwidth_monitoring,
        edx,
        1
    );

    check_bit_fn!(
        doc = "Supports local bandwidth monitoring.",
        has_local_bandwidth_monitoring,
        edx,
        2
    );
}

impl Debug for L3MonitoringInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("L3MonitoringInfo")
            .field("conversion_factor", &self.conversion_factor())
            .field("maximum_rmid_range", &self.maximum_rmid_range())
            .finish()
    }
}

/// Quality of service enforcement information (LEAF=0x10).
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct RdtAllocationInfo<R: CpuIdReader> {
    read: R,
    ebx: u32,
}

impl<R: CpuIdReader> RdtAllocationInfo<R> {
    check_bit_fn!(doc = "Supports L3 Cache Allocation.", has_l3_cat, ebx, 1);

    check_bit_fn!(doc = "Supports L2 Cache Allocation.", has_l2_cat, ebx, 2);

    check_bit_fn!(
        doc = "Supports Memory Bandwidth Allocation.",
        has_memory_bandwidth_allocation,
        ebx,
        3
    );

    /// L3 Cache Allocation Information.
    pub fn l3_cat(&self) -> Option<L3CatInfo> {
        if self.has_l3_cat() {
            let res = self.read.cpuid2(EAX_RDT_ALLOCATION, 1);
            Some(L3CatInfo {
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// L2 Cache Allocation Information.
    pub fn l2_cat(&self) -> Option<L2CatInfo> {
        if self.has_l2_cat() {
            let res = self.read.cpuid2(EAX_RDT_ALLOCATION, 2);
            Some(L2CatInfo {
                eax: res.eax,
                ebx: res.ebx,
                edx: res.edx,
            })
        } else {
            None
        }
    }

    /// Memory Bandwidth Allocation Information.
    pub fn memory_bandwidth_allocation(&self) -> Option<MemBwAllocationInfo> {
        if self.has_memory_bandwidth_allocation() {
            let res = self.read.cpuid2(EAX_RDT_ALLOCATION, 3);
            Some(MemBwAllocationInfo {
                eax: res.eax,
                ecx: res.ecx,
                edx: res.edx,
            })
        } else {
            None
        }
    }
}

impl<R: CpuIdReader> Debug for RdtAllocationInfo<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RdtAllocationInfo")
            .field("l3_cat", &self.l3_cat())
            .field("l2_cat", &self.l2_cat())
            .field(
                "memory_bandwidth_allocation",
                &self.memory_bandwidth_allocation(),
            )
            .finish()
    }
}

/// L3 Cache Allocation Technology Enumeration Sub-leaf (LEAF=0x10, SUBLEAF=1).
pub struct L3CatInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl L3CatInfo {
    /// Length of the capacity bit mask.
    pub fn capacity_mask_length(&self) -> u8 {
        (get_bits(self.eax, 0, 4) + 1) as u8
    }

    /// Bit-granular map of isolation/contention of allocation units.
    pub fn isolation_bitmap(&self) -> u32 {
        self.ebx
    }

    /// Highest COS number supported for this Leaf.
    pub fn highest_cos(&self) -> u16 {
        get_bits(self.edx, 0, 15) as u16
    }

    check_bit_fn!(
        doc = "Is Code and Data Prioritization Technology supported?",
        has_code_data_prioritization,
        ecx,
        2
    );
}

impl Debug for L3CatInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("L3CatInfo")
            .field("capacity_mask_length", &self.capacity_mask_length())
            .field("isolation_bitmap", &self.isolation_bitmap())
            .field("highest_cos", &self.highest_cos())
            .finish()
    }
}

/// L2 Cache Allocation Technology Enumeration Sub-leaf (LEAF=0x10, SUBLEAF=2).
#[derive(Eq, PartialEq)]
pub struct L2CatInfo {
    eax: u32,
    ebx: u32,
    edx: u32,
}

impl L2CatInfo {
    /// Length of the capacity bit mask.
    pub fn capacity_mask_length(&self) -> u8 {
        (get_bits(self.eax, 0, 4) + 1) as u8
    }

    /// Bit-granular map of isolation/contention of allocation units.
    pub fn isolation_bitmap(&self) -> u32 {
        self.ebx
    }

    /// Highest COS number supported for this Leaf.
    pub fn highest_cos(&self) -> u16 {
        get_bits(self.edx, 0, 15) as u16
    }
}

impl Debug for L2CatInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("L2CatInfo")
            .field("capacity_mask_length", &self.capacity_mask_length())
            .field("isolation_bitmap", &self.isolation_bitmap())
            .field("highest_cos", &self.highest_cos())
            .finish()
    }
}

/// Memory Bandwidth Allocation Enumeration Sub-leaf (LEAF=0x10, SUBLEAF=3).
#[derive(Eq, PartialEq)]
pub struct MemBwAllocationInfo {
    eax: u32,
    ecx: u32,
    edx: u32,
}

impl MemBwAllocationInfo {
    /// Reports the maximum MBA throttling value supported for the corresponding ResID.
    pub fn max_hba_throttling(&self) -> u16 {
        (get_bits(self.eax, 0, 11) + 1) as u16
    }

    /// Highest COS number supported for this Leaf.
    pub fn highest_cos(&self) -> u16 {
        get_bits(self.edx, 0, 15) as u16
    }

    check_bit_fn!(
        doc = "Reports whether the response of the delay values is linear.",
        has_linear_response_delay,
        ecx,
        2
    );
}

impl Debug for MemBwAllocationInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MemBwAllocationInfo")
            .field("max_hba_throttling", &self.max_hba_throttling())
            .field("highest_cos", &self.highest_cos())
            .field(
                "has_linear_response_delay",
                &self.has_linear_response_delay(),
            )
            .finish()
    }
}

/// Intel SGX Capability Enumeration Leaf (LEAF=0x12).
///
/// Two sub-leafs: (EAX = 12H, ECX = 0 and ECX = 1)
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct SgxInfo<R: CpuIdReader> {
    read: R,
    eax: u32,
    ebx: u32,
    _ecx: u32,
    edx: u32,
    eax1: u32,
    ebx1: u32,
    ecx1: u32,
    edx1: u32,
}

impl<F: CpuIdReader> SgxInfo<F> {
    check_bit_fn!(doc = "Has SGX1 support.", has_sgx1, eax, 0);
    check_bit_fn!(doc = "Has SGX2 support.", has_sgx2, eax, 1);

    check_bit_fn!(
        doc = "Supports ENCLV instruction leaves EINCVIRTCHILD, EDECVIRTCHILD, and ESETCONTEXT.",
        has_enclv_leaves_einvirtchild_edecvirtchild_esetcontext,
        eax,
        5
    );

    check_bit_fn!(
        doc = "Supports ENCLS instruction leaves ETRACKC, ERDINFO, ELDBC, and ELDUC.",
        has_encls_leaves_etrackc_erdinfo_eldbc_elduc,
        eax,
        6
    );

    /// Bit vector of supported extended SGX features.
    pub fn miscselect(&self) -> u32 {
        self.ebx
    }

    ///  The maximum supported enclave size in non-64-bit mode is 2^retval.
    pub fn max_enclave_size_non_64bit(&self) -> u8 {
        get_bits(self.edx, 0, 7) as u8
    }

    ///  The maximum supported enclave size in 64-bit mode is 2^retval.
    pub fn max_enclave_size_64bit(&self) -> u8 {
        get_bits(self.edx, 8, 15) as u8
    }

    /// Reports the valid bits of SECS.ATTRIBUTES\[127:0\] that software can set with ECREATE.
    pub fn secs_attributes(&self) -> (u64, u64) {
        let lower = self.eax1 as u64 | (self.ebx1 as u64) << 32;
        let upper = self.ecx1 as u64 | (self.edx1 as u64) << 32;
        (lower, upper)
    }
    /// Iterator over SGX sub-leafs.
    pub fn iter(&self) -> SgxSectionIter<F> {
        SgxSectionIter {
            read: self.read.clone(),
            current: 2,
        }
    }
}

impl<R: CpuIdReader> Debug for SgxInfo<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SgxInfo")
            .field("has_sgx1", &self.has_sgx1())
            .field("has_sgx2", &self.has_sgx2())
            .field("miscselect", &self.miscselect())
            .field(
                "max_enclave_size_non_64bit",
                &self.max_enclave_size_non_64bit(),
            )
            .field("max_enclave_size_64bit", &self.max_enclave_size_64bit())
            .field(
                "has_encls_leaves_etrackc_erdinfo_eldbc_elduc",
                &self.has_encls_leaves_etrackc_erdinfo_eldbc_elduc(),
            )
            .field(
                "has_enclv_leaves_einvirtchild_edecvirtchild_esetcontext",
                &self.has_enclv_leaves_einvirtchild_edecvirtchild_esetcontext(),
            )
            .field("sgx_section_iter", &self.iter())
            .finish()
    }
}

/// Iterator over the SGX sub-leafs (ECX >= 2).
#[derive(Clone)]
pub struct SgxSectionIter<R: CpuIdReader> {
    read: R,
    current: u32,
}

impl<R: CpuIdReader> Iterator for SgxSectionIter<R> {
    type Item = SgxSectionInfo;

    fn next(&mut self) -> Option<SgxSectionInfo> {
        let res = self.read.cpuid2(EAX_SGX, self.current);
        self.current += 1;
        match get_bits(res.eax, 0, 3) {
            0b0001 => Some(SgxSectionInfo::Epc(EpcSection {
                eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            })),
            _ => None,
        }
    }
}

impl<R: CpuIdReader> Debug for SgxSectionIter<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

/// Intel SGX EPC Enumeration Leaf
///
/// Sub-leaves 2 or higher.
#[derive(Debug)]
pub enum SgxSectionInfo {
    // This would be nice: https://github.com/rust-lang/rfcs/pull/1450
    Epc(EpcSection),
}

/// EBX:EAX and EDX:ECX provide information on the Enclave Page Cache (EPC) section
#[derive(Debug)]
pub struct EpcSection {
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl EpcSection {
    /// The physical address of the base of the EPC section
    pub fn physical_base(&self) -> u64 {
        let lower = (get_bits(self.eax, 12, 31) << 12) as u64;
        let upper = (get_bits(self.ebx, 0, 19) as u64) << 32;
        lower | upper
    }

    /// Size of the corresponding EPC section within the Processor Reserved Memory.
    pub fn size(&self) -> u64 {
        let lower = (get_bits(self.ecx, 12, 31) << 12) as u64;
        let upper = (get_bits(self.edx, 0, 19) as u64) << 32;
        lower | upper
    }
}

/// Intel Processor Trace Information (LEAF=0x14).
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct ProcessorTraceInfo {
    _eax: u32,
    ebx: u32,
    ecx: u32,
    _edx: u32,
    leaf1: Option<CpuIdResult>,
}

impl ProcessorTraceInfo {
    // EBX features
    check_bit_fn!(
        doc = "If true, Indicates that IA32_RTIT_CTL.CR3Filter can be set to 1, and \
               that IA32_RTIT_CR3_MATCH MSR can be accessed.",
        has_rtit_cr3_match,
        ebx,
        0
    );
    check_bit_fn!(
        doc = "If true, Indicates support of Configurable PSB and Cycle-Accurate Mode.",
        has_configurable_psb_and_cycle_accurate_mode,
        ebx,
        1
    );
    check_bit_fn!(
        doc = "If true, Indicates support of IP Filtering, TraceStop filtering, and \
               preservation of Intel PT MSRs across warm reset.",
        has_ip_tracestop_filtering,
        ebx,
        2
    );
    check_bit_fn!(
        doc = "If true, Indicates support of MTC timing packet and suppression of \
               COFI-based packets.",
        has_mtc_timing_packet_coefi_suppression,
        ebx,
        3
    );

    check_bit_fn!(
        doc = "Indicates support of PTWRITE. Writes can set IA32_RTIT_CTL\\[12\\] (PTWEn \
               and IA32_RTIT_CTL\\[5\\] (FUPonPTW), and PTWRITE can generate packets",
        has_ptwrite,
        ebx,
        4
    );

    check_bit_fn!(
        doc = "Support of Power Event Trace. Writes can set IA32_RTIT_CTL\\[4\\] (PwrEvtEn) \
               enabling Power Event Trace packet generation.",
        has_power_event_trace,
        ebx,
        5
    );

    // ECX features
    check_bit_fn!(
        doc = "If true, Tracing can be enabled with IA32_RTIT_CTL.ToPA = 1, hence \
               utilizing the ToPA output scheme; IA32_RTIT_OUTPUT_BASE and \
               IA32_RTIT_OUTPUT_MASK_PTRS MSRs can be accessed.",
        has_topa,
        ecx,
        0
    );
    check_bit_fn!(
        doc = "If true, ToPA tables can hold any number of output entries, up to the \
               maximum allowed by the MaskOrTableOffset field of \
               IA32_RTIT_OUTPUT_MASK_PTRS.",
        has_topa_maximum_entries,
        ecx,
        1
    );
    check_bit_fn!(
        doc = "If true, Indicates support of Single-Range Output scheme.",
        has_single_range_output_scheme,
        ecx,
        2
    );
    check_bit_fn!(
        doc = "If true, Indicates support of output to Trace Transport subsystem.",
        has_trace_transport_subsystem,
        ecx,
        3
    );
    check_bit_fn!(
        doc = "If true, Generated packets which contain IP payloads have LIP values, \
               which include the CS base component.",
        has_lip_with_cs_base,
        ecx,
        31
    );

    /// Number of configurable Address Ranges for filtering (Bits 2:0).
    pub fn configurable_address_ranges(&self) -> u8 {
        self.leaf1.map_or(0, |res| get_bits(res.eax, 0, 2) as u8)
    }

    /// Bitmap of supported MTC period encodings (Bit 31:16).
    pub fn supported_mtc_period_encodings(&self) -> u16 {
        self.leaf1.map_or(0, |res| get_bits(res.eax, 16, 31) as u16)
    }

    /// Bitmap of supported Cycle Threshold value encodings (Bits 15-0).
    pub fn supported_cycle_threshold_value_encodings(&self) -> u16 {
        self.leaf1.map_or(0, |res| get_bits(res.ebx, 0, 15) as u16)
    }

    /// Bitmap of supported Configurable PSB frequency encodings (Bit 31:16)
    pub fn supported_psb_frequency_encodings(&self) -> u16 {
        self.leaf1.map_or(0, |res| get_bits(res.ebx, 16, 31) as u16)
    }
}

impl Debug for ProcessorTraceInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProcessorTraceInfo")
            .field(
                "configurable_address_ranges",
                &self.configurable_address_ranges(),
            )
            .field(
                "supported_mtc_period_encodings",
                &self.supported_mtc_period_encodings(),
            )
            .field(
                "supported_cycle_threshold_value_encodings",
                &self.supported_cycle_threshold_value_encodings(),
            )
            .field(
                "supported_psb_frequency_encodings",
                &self.supported_psb_frequency_encodings(),
            )
            .finish()
    }
}

/// Time Stamp Counter/Core Crystal Clock Information (LEAF=0x15).
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct TscInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
}

impl fmt::Debug for TscInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TscInfo")
            .field("denominator", &self.denominator())
            .field("numerator", &self.numerator())
            .field("nominal_frequency", &self.nominal_frequency())
            .field("tsc_frequency", &self.tsc_frequency())
            .finish()
    }
}

impl TscInfo {
    /// An unsigned integer which is the denominator of the TSC/”core crystal clock” ratio.
    pub fn denominator(&self) -> u32 {
        self.eax
    }

    /// An unsigned integer which is the numerator of the TSC/”core crystal clock” ratio.
    ///
    /// If this is 0, the TSC/”core crystal clock” ratio is not enumerated.
    pub fn numerator(&self) -> u32 {
        self.ebx
    }

    /// An unsigned integer which is the nominal frequency of the core crystal clock in Hz.
    ///
    /// If this is 0, the nominal core crystal clock frequency is not enumerated.
    pub fn nominal_frequency(&self) -> u32 {
        self.ecx
    }

    /// “TSC frequency” = “core crystal clock frequency” * EBX/EAX.
    pub fn tsc_frequency(&self) -> Option<u64> {
        // In some case TscInfo is a valid leaf, but the values reported are still 0
        // we should avoid a division by zero in case denominator ends up being 0.
        if self.nominal_frequency() == 0 || self.numerator() == 0 || self.denominator() == 0 {
            return None;
        }

        Some(self.nominal_frequency() as u64 * self.numerator() as u64 / self.denominator() as u64)
    }
}

/// Processor Frequency Information (LEAF=0x16).
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct ProcessorFrequencyInfo {
    eax: u32,
    ebx: u32,
    ecx: u32,
}

impl ProcessorFrequencyInfo {
    /// Processor Base Frequency (in MHz).
    pub fn processor_base_frequency(&self) -> u16 {
        get_bits(self.eax, 0, 15) as u16
    }

    /// Maximum Frequency (in MHz).
    pub fn processor_max_frequency(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    /// Bus (Reference) Frequency (in MHz).
    pub fn bus_frequency(&self) -> u16 {
        get_bits(self.ecx, 0, 15) as u16
    }
}

impl fmt::Debug for ProcessorFrequencyInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ProcessorFrequencyInfo")
            .field("processor_base_frequency", &self.processor_base_frequency())
            .field("processor_max_frequency", &self.processor_max_frequency())
            .field("bus_frequency", &self.bus_frequency())
            .finish()
    }
}

/// Deterministic Address Translation Structure Iterator (LEAF=0x18).
///
/// # Platforms
/// ❌ AMD ✅ Intel
#[derive(Clone)]
pub struct DatIter<R: CpuIdReader> {
    read: R,
    current: u32,
    count: u32,
}

impl<R: CpuIdReader> Iterator for DatIter<R> {
    type Item = DatInfo;

    /// Iterate over each sub-leaf with an address translation structure.
    fn next(&mut self) -> Option<DatInfo> {
        loop {
            // Sub-leaf index n is invalid if n exceeds the value that sub-leaf 0 returns in EAX
            if self.current > self.count {
                return None;
            }

            let res = self
                .read
                .cpuid2(EAX_DETERMINISTIC_ADDRESS_TRANSLATION_INFO, self.current);
            self.current += 1;

            // A sub-leaf index is also invalid if EDX[4:0] returns 0.
            if get_bits(res.edx, 0, 4) == 0 {
                // Valid sub-leaves do not need to be contiguous or in any particular order.
                // A valid sub-leaf may be in a higher input ECX value than an invalid sub-leaf
                // or than a valid sub-leaf of a higher or lower-level struc-ture
                continue;
            }

            return Some(DatInfo {
                _eax: res.eax,
                ebx: res.ebx,
                ecx: res.ecx,
                edx: res.edx,
            });
        }
    }
}

impl<R: CpuIdReader> Debug for DatIter<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        self.clone().for_each(|ref item| {
            debug.entry(item);
        });
        debug.finish()
    }
}

/// Deterministic Address Translation Structure
pub struct DatInfo {
    _eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl DatInfo {
    check_bit_fn!(
        doc = "4K page size entries supported by this structure",
        has_4k_entries,
        ebx,
        0
    );

    check_bit_fn!(
        doc = "2MB page size entries supported by this structure",
        has_2mb_entries,
        ebx,
        1
    );

    check_bit_fn!(
        doc = "4MB page size entries supported by this structure",
        has_4mb_entries,
        ebx,
        2
    );

    check_bit_fn!(
        doc = "1GB page size entries supported by this structure",
        has_1gb_entries,
        ebx,
        3
    );

    check_bit_fn!(
        doc = "Fully associative structure",
        is_fully_associative,
        edx,
        8
    );

    /// Partitioning (0: Soft partitioning between the logical processors sharing this structure).
    pub fn partitioning(&self) -> u8 {
        get_bits(self.ebx, 8, 10) as u8
    }

    /// Ways of associativity.
    pub fn ways(&self) -> u16 {
        get_bits(self.ebx, 16, 31) as u16
    }

    /// Number of Sets.
    pub fn sets(&self) -> u32 {
        self.ecx
    }

    /// Translation cache type field.
    pub fn cache_type(&self) -> DatType {
        match get_bits(self.edx, 0, 4) as u8 {
            0b00001 => DatType::DataTLB,
            0b00010 => DatType::InstructionTLB,
            0b00011 => DatType::UnifiedTLB,
            0b00000 => DatType::Null, // should never be returned as this indicates invalid struct!
            0b00100 => DatType::LoadOnly,
            0b00101 => DatType::StoreOnly,
            _ => DatType::Unknown,
        }
    }

    /// Translation cache level (starts at 1)
    pub fn cache_level(&self) -> u8 {
        get_bits(self.edx, 5, 7) as u8
    }

    /// Maximum number of addressable IDs for logical processors sharing this translation cache
    pub fn max_addressable_ids(&self) -> u16 {
        // Add one to the return value to get the result:
        (get_bits(self.edx, 14, 25) + 1) as u16
    }
}

impl Debug for DatInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DatInfo")
            .field("has_4k_entries", &self.has_4k_entries())
            .field("has_2mb_entries", &self.has_2mb_entries())
            .field("has_4mb_entries", &self.has_4mb_entries())
            .field("has_1gb_entries", &self.has_1gb_entries())
            .field("is_fully_associative", &self.is_fully_associative())
            .finish()
    }
}

/// Deterministic Address Translation cache type (EDX bits 04 -- 00)
#[derive(Eq, PartialEq, Debug)]
pub enum DatType {
    /// Null (indicates this sub-leaf is not valid).
    Null = 0b00000,
    DataTLB = 0b00001,
    InstructionTLB = 0b00010,
    /// Some unified TLBs will allow a single TLB entry to satisfy data read/write
    /// and instruction fetches. Others will require separate entries (e.g., one
    /// loaded on data read/write and another loaded on an instruction fetch) .
    /// Please see the Intel® 64 and IA-32 Architectures Optimization Reference Manual
    /// for details of a particular product.
    UnifiedTLB = 0b00011,
    LoadOnly = 0b0100,
    StoreOnly = 0b0101,
    Unknown,
}

impl fmt::Display for DatType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let t = match self {
            DatType::Null => "invalid (0)",
            DatType::DataTLB => "Data TLB",
            DatType::InstructionTLB => "Instruction TLB",
            DatType::UnifiedTLB => "Unified TLB",
            DatType::LoadOnly => "Load Only",
            DatType::StoreOnly => "Store Only",
            DatType::Unknown => "Unknown",
        };
        f.write_str(t)
    }
}

/// SoC vendor specific information (LEAF=0x17).
///
/// # Platforms
/// ❌ AMD ✅ Intel
pub struct SoCVendorInfo<R: CpuIdReader> {
    read: R,
    /// MaxSOCID_Index
    eax: u32,
    ebx: u32,
    ecx: u32,
    edx: u32,
}

impl<R: CpuIdReader> SoCVendorInfo<R> {
    pub fn get_soc_vendor_id(&self) -> u16 {
        get_bits(self.ebx, 0, 15) as u16
    }

    pub fn get_project_id(&self) -> u32 {
        self.ecx
    }

    pub fn get_stepping_id(&self) -> u32 {
        self.edx
    }

    pub fn get_vendor_brand(&self) -> Option<SoCVendorBrand> {
        // Leaf 17H is valid if MaxSOCID_Index >= 3.
        if self.eax >= 3 {
            let r1 = self.read.cpuid2(EAX_SOC_VENDOR_INFO, 1);
            let r2 = self.read.cpuid2(EAX_SOC_VENDOR_INFO, 2);
            let r3 = self.read.cpuid2(EAX_SOC_VENDOR_INFO, 3);
            Some(SoCVendorBrand { data: [r1, r2, r3] })
        } else {
            None
        }
    }

    pub fn get_vendor_attributes(&self) -> Option<SoCVendorAttributesIter<R>> {
        if self.eax > 3 {
            Some(SoCVendorAttributesIter {
                read: self.read.clone(),
                count: self.eax,
                current: 3,
            })
        } else {
            None
        }
    }
}

impl<R: CpuIdReader> fmt::Debug for SoCVendorInfo<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SoCVendorInfo")
            .field("soc_vendor_id", &self.get_soc_vendor_id())
            .field("project_id", &self.get_project_id())
            .field("stepping_id", &self.get_stepping_id())
            .field("vendor_brand", &self.get_vendor_brand())
            .field("vendor_attributes", &self.get_vendor_attributes())
            .finish()
    }
}

/// Iterator for SoC vendor attributes.
pub struct SoCVendorAttributesIter<R: CpuIdReader> {
    read: R,
    count: u32,
    current: u32,
}

impl<R: CpuIdReader> fmt::Debug for SoCVendorAttributesIter<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SocVendorAttributesIter")
            .field("count", &self.count)
            .field("current", &self.current)
            .finish()
    }
}

impl<R: CpuIdReader> Iterator for SoCVendorAttributesIter<R> {
    type Item = CpuIdResult;

    /// Iterate over all SoC vendor specific attributes.
    fn next(&mut self) -> Option<CpuIdResult> {
        if self.current > self.count {
            return None;
        }
        self.count += 1;
        Some(self.read.cpuid2(EAX_SOC_VENDOR_INFO, self.count))
    }
}

/// A vendor brand string as queried from the cpuid leaf.
#[derive(Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SoCVendorBrand {
    data: [CpuIdResult; 3],
}

impl SoCVendorBrand {
    /// Return the SocVendorBrand as a string.
    pub fn as_str(&self) -> &str {
        let brand_string_start = self as *const SoCVendorBrand as *const u8;
        let slice = unsafe {
            // Safety: SoCVendorBrand is laid out with repr(C).
            slice::from_raw_parts(brand_string_start, size_of::<SoCVendorBrand>())
        };
        str::from_utf8(slice).unwrap_or("InvalidSoCVendorString")
    }

    #[deprecated(
        since = "10.0.0",
        note = "Use idiomatic function name `as_str` instead"
    )]
    pub fn as_string(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for SoCVendorBrand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Information about Hypervisor (LEAF=0x4000_0001)
///
/// More information about this semi-official leaf can be found here
/// <https://lwn.net/Articles/301888/>
pub struct HypervisorInfo<R: CpuIdReader> {
    read: R,
    res: CpuIdResult,
}

impl<R: CpuIdReader> fmt::Debug for HypervisorInfo<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HypervisorInfo")
            .field("identify", &self.identify())
            .field("tsc_frequency", &self.tsc_frequency())
            .field("apic_frequency", &self.apic_frequency())
            .finish()
    }
}

/// Identifies the different Hypervisor products.
#[derive(Debug, Eq, PartialEq)]
pub enum Hypervisor {
    Xen,
    VMware,
    HyperV,
    KVM,
    /// QEMU is the hypervisor identity when QEMU is used
    /// without an accelerator, such as KVM.
    QEMU,
    Bhyve,
    QNX,
    ACRN,
    Unknown(u32, u32, u32),
}

impl<R: CpuIdReader> HypervisorInfo<R> {
    /// Returns the identity of the [`Hypervisor`].
    ///
    /// ## Technical Background
    ///
    /// The value is a 12-byte (12 character) fixed-length ASCII string.
    ///
    /// Usually all of these IDs can be found in the original source code on
    /// Github relatively easy (if the project is open source). Once you
    /// have an ID, you find cumulated lists with all kinds of IDs on Github
    /// relatively easy.
    pub fn identify(&self) -> Hypervisor {
        match (self.res.ebx, self.res.ecx, self.res.edx) {
            // "VMwareVMware" (0x56 => V, 0x4d => M, ...)
            (0x61774d56, 0x4d566572, 0x65726177) => Hypervisor::VMware,
            // "XenVMMXenVMM"
            (0x566e6558, 0x65584d4d, 0x4d4d566e) => Hypervisor::Xen,
            // "Microsoft Hv"
            (0x7263694d, 0x666f736f, 0x76482074) => Hypervisor::HyperV,
            // "KVMKVMKVM\0\0\0"
            (0x4b4d564b, 0x564b4d56, 0x0000004d) => Hypervisor::KVM,
            // "TCGTCGTCGTCG"
            // see https://github.com/qemu/qemu/blob/6512fa497c2fa9751b9d774ab32d87a9764d1958/target/i386/cpu.c
            (0x54474354, 0x43544743, 0x47435447) => Hypervisor::QEMU,
            // "bhyve bhyve "
            // found this in another library ("heim-virt")
            (0x76796862, 0x68622065, 0x20657679) => Hypervisor::Bhyve,
            // "BHyVE BHyVE "
            // But this value is in the original source code. To be safe, we keep both.
            // See https://github.com/lattera/bhyve/blob/5946a9115d2771a1d27f14a835c7fbc05b30f7f9/sys/amd64/vmm/x86.c#L165
            (0x56794842, 0x48422045, 0x20455679) => Hypervisor::Bhyve,
            // "QNXQVMBSQG"
            // This can be verified in multiple Git repos (e.g. by Intel)
            // https://github.com/search?q=QNXQVMBSQG&type=code
            (0x51584e51, 0x53424d56, 0x00004751) => Hypervisor::QNX,
            // "ACRNACRNACRN"
            (0x4e524341, 0x4e524341, 0x4e524341) => Hypervisor::ACRN,
            (ebx, ecx, edx) => Hypervisor::Unknown(ebx, ecx, edx),
        }
    }

    /// TSC frequency in kHz.
    pub fn tsc_frequency(&self) -> Option<u32> {
        // vm aware tsc frequency retrieval:
        // # EAX: (Virtual) TSC frequency in kHz.
        if self.res.eax >= 0x40000010 {
            let virt_tinfo = self.read.cpuid2(0x40000010, 0);
            Some(virt_tinfo.eax)
        } else {
            None
        }
    }

    /// (Virtual) Bus (local apic timer) frequency in kHz.
    pub fn apic_frequency(&self) -> Option<u32> {
        // # EBX: (Virtual) Bus (local apic timer) frequency in kHz.
        if self.res.eax >= 0x40000010 {
            let virt_tinfo = self.read.cpuid2(0x40000010, 0);
            Some(virt_tinfo.ebx)
        } else {
            None
        }
    }
}

#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }

    external_doc_test!(include_str!("../README.md"));
}
