use crate::{Associativity, CpuId, CpuIdResult, TopologyType};
use phf::phf_map;

/// Raw dump of ryzen mantisse cpuid values.
///
/// Key format is (eax << 32 | ecx) e.g., two 32 bit values packed in one 64 bit value
///
///
/// # Representation of Hex Values
///
/// ```log
///   vendor_id = "AuthenticAMD"
///   version information (1/eax):
///      processor type  = primary processor (0)
///      family          = 0xf (15)
///      model           = 0x1 (1)
///      stepping id     = 0x0 (0)
///      extended family = 0x8 (8)
///      extended model  = 0x7 (7)
///      (family synth)  = 0x17 (23)
///      (model synth)   = 0x71 (113)
///      (simple synth)  = AMD Ryzen (Matisse B0) [Zen 2], 7nm
///   miscellaneous (1/ebx):
///      process local APIC physical ID = 0xa (10)
///      cpu count                      = 0xc (12)
///      CLFLUSH line size              = 0x8 (8)
///      brand index                    = 0x0 (0)
///   brand id = 0x00 (0): unknown
///   feature information (1/edx):
///      x87 FPU on chip                        = true
///      VME: virtual-8086 mode enhancement     = true
///      DE: debugging extensions               = true
///      PSE: page size extensions              = true
///      TSC: time stamp counter                = true
///      RDMSR and WRMSR support                = true
///      PAE: physical address extensions       = true
///      MCE: machine check exception           = true
///      CMPXCHG8B inst.                        = true
///      APIC on chip                           = true
///      SYSENTER and SYSEXIT                   = true
///      MTRR: memory type range registers      = true
///      PTE global bit                         = true
///      MCA: machine check architecture        = true
///      CMOV: conditional move/compare instr   = true
///      PAT: page attribute table              = true
///      PSE-36: page size extension            = true
///      PSN: processor serial number           = false
///      CLFLUSH instruction                    = true
///      DS: debug store                        = false
///      ACPI: thermal monitor and clock ctrl   = false
///      MMX Technology                         = true
///      FXSAVE/FXRSTOR                         = true
///      SSE extensions                         = true
///      SSE2 extensions                        = true
///      SS: self snoop                         = false
///      hyper-threading / multi-core supported = true
///      TM: therm. monitor                     = false
///      IA64                                   = false
///      PBE: pending break event               = false
///   feature information (1/ecx):
///      PNI/SSE3: Prescott New Instructions     = true
///      PCLMULDQ instruction                    = true
///      DTES64: 64-bit debug store              = false
///      MONITOR/MWAIT                           = true
///      CPL-qualified debug store               = false
///      VMX: virtual machine extensions         = false
///      SMX: safer mode extensions              = false
///      Enhanced Intel SpeedStep Technology     = false
///      TM2: thermal monitor 2                  = false
///      SSSE3 extensions                        = true
///      context ID: adaptive or shared L1 data  = false
///      SDBG: IA32_DEBUG_INTERFACE              = false
///      FMA instruction                         = true
///      CMPXCHG16B instruction                  = true
///      xTPR disable                            = false
///      PDCM: perfmon and debug                 = false
///      PCID: process context identifiers       = false
///      DCA: direct cache access                = false
///      SSE4.1 extensions                       = true
///      SSE4.2 extensions                       = true
///      x2APIC: extended xAPIC support          = false
///      MOVBE instruction                       = true
///      POPCNT instruction                      = true
///      time stamp counter deadline             = false
///      AES instruction                         = true
///      XSAVE/XSTOR states                      = true
///      OS-enabled XSAVE/XSTOR                  = true
///      AVX: advanced vector extensions         = true
///      F16C half-precision convert instruction = true
///      RDRAND instruction                      = true
///      hypervisor guest status                 = false
///   cache and TLB information (2):
///   processor serial number = 0087-0F10-0000-0000-0000-0000
///   MONITOR/MWAIT (5):
///      smallest monitor-line size (bytes)       = 0x40 (64)
///      largest monitor-line size (bytes)        = 0x40 (64)
///      enum of Monitor-MWAIT exts supported     = true
///      supports intrs as break-event for MWAIT  = true
///      number of C0 sub C-states using MWAIT    = 0x1 (1)
///      number of C1 sub C-states using MWAIT    = 0x1 (1)
///      number of C2 sub C-states using MWAIT    = 0x0 (0)
///      number of C3 sub C-states using MWAIT    = 0x0 (0)
///      number of C4 sub C-states using MWAIT    = 0x0 (0)
///      number of C5 sub C-states using MWAIT    = 0x0 (0)
///      number of C6 sub C-states using MWAIT    = 0x0 (0)
///      number of C7 sub C-states using MWAIT    = 0x0 (0)
///   Thermal and Power Management Features (6):
///      digital thermometer                     = false
///      Intel Turbo Boost Technology            = false
///      ARAT always running APIC timer          = true
///      PLN power limit notification            = false
///      ECMD extended clock modulation duty     = false
///      PTM package thermal management          = false
///      HWP base registers                      = false
///      HWP notification                        = false
///      HWP activity window                     = false
///      HWP energy performance preference       = false
///      HWP package level request               = false
///      HDC base registers                      = false
///      Intel Turbo Boost Max Technology 3.0    = false
///      HWP capabilities                        = false
///      HWP PECI override                       = false
///      flexible HWP                            = false
///      IA32_HWP_REQUEST MSR fast access mode   = false
///      HW_FEEDBACK                             = false
///      ignoring idle logical processor HWP req = false
///      digital thermometer thresholds          = 0x0 (0)
///      hardware coordination feedback          = true
///      ACNT2 available                         = false
///      performance-energy bias capability      = false
///      performance capability reporting        = false
///      energy efficiency capability reporting  = false
///      size of feedback struct (4KB pages)     = 0x0 (0)
///      index of CPU's row in feedback struct   = 0x0 (0)
///   extended feature flags (7):
///      FSGSBASE instructions                    = true
///      IA32_TSC_ADJUST MSR supported            = false
///      SGX: Software Guard Extensions supported = false
///      BMI1 instructions                        = true
///      HLE hardware lock elision                = false
///      AVX2: advanced vector extensions 2       = true
///      FDP_EXCPTN_ONLY                          = false
///      SMEP supervisor mode exec protection     = true
///      BMI2 instructions                        = true
///      enhanced REP MOVSB/STOSB                 = false
///      INVPCID instruction                      = false
///      RTM: restricted transactional memory     = false
///      RDT-CMT/PQoS cache monitoring            = true
///      deprecated FPU CS/DS                     = false
///      MPX: intel memory protection extensions  = false
///      RDT-CAT/PQE cache allocation             = true
///      AVX512F: AVX-512 foundation instructions = false
///      AVX512DQ: double & quadword instructions = false
///      RDSEED instruction                       = true
///      ADX instructions                         = true
///      SMAP: supervisor mode access prevention  = true
///      AVX512IFMA: fused multiply add           = false
///      PCOMMIT instruction                      = false
///      CLFLUSHOPT instruction                   = true
///      CLWB instruction                         = true
///      Intel processor trace                    = false
///      AVX512PF: prefetch instructions          = false
///      AVX512ER: exponent & reciprocal instrs   = false
///      AVX512CD: conflict detection instrs      = false
///      SHA instructions                         = true
///      AVX512BW: byte & word instructions       = false
///      AVX512VL: vector length                  = false
///      PREFETCHWT1                              = false
///      AVX512VBMI: vector byte manipulation     = false
///      UMIP: user-mode instruction prevention   = true
///      PKU protection keys for user-mode        = false
///      OSPKE CR4.PKE and RDPKRU/WRPKRU          = false
///      WAITPKG instructions                     = false
///      AVX512_VBMI2: byte VPCOMPRESS, VPEXPAND  = false
///      CET_SS: CET shadow stack                 = false
///      GFNI: Galois Field New Instructions      = false
///      VAES instructions                        = false
///      VPCLMULQDQ instruction                   = false
///      AVX512_VNNI: neural network instructions = false
///      AVX512_BITALG: bit count/shiffle         = false
///      TME: Total Memory Encryption             = false
///      AVX512: VPOPCNTDQ instruction            = false
///      5-level paging                           = false
///      BNDLDX/BNDSTX MAWAU value in 64-bit mode = 0x0 (0)
///      RDPID: read processor D supported        = true
///      CLDEMOTE supports cache line demote      = false
///      MOVDIRI instruction                      = false
///      MOVDIR64B instruction                    = false
///      ENQCMD instruction                       = false
///      SGX_LC: SGX launch config supported      = false
///      AVX512_4VNNIW: neural network instrs     = false
///      AVX512_4FMAPS: multiply acc single prec  = false
///      fast short REP MOV                       = false
///      AVX512_VP2INTERSECT: intersect mask regs = false
///      VERW md-clear microcode support          = false
///      hybrid part                              = false
///      PCONFIG instruction                      = false
///      CET_IBT: CET indirect branch tracking    = false
///      IBRS/IBPB: indirect branch restrictions  = false
///      STIBP: 1 thr indirect branch predictor   = false
///      L1D_FLUSH: IA32_FLUSH_CMD MSR            = false
///      IA32_ARCH_CAPABILITIES MSR               = false
///      IA32_CORE_CAPABILITIES MSR               = false
///      SSBD: speculative store bypass disable   = false
///   Direct Cache Access Parameters (9):
///      PLATFORM_DCA_CAP MSR bits = 0
///   Architecture Performance Monitoring Features (0xa/eax):
///      version ID                               = 0x0 (0)
///      number of counters per logical processor = 0x0 (0)
///      bit width of counter                     = 0x0 (0)
///      length of EBX bit vector                 = 0x0 (0)
///   Architecture Performance Monitoring Features (0xa/ebx):
///      core cycle event not available           = false
///      instruction retired event not available  = false
///      reference cycles event not available     = false
///      last-level cache ref event not available = false
///      last-level cache miss event not avail    = false
///      branch inst retired event not available  = false
///      branch mispred retired event not avail   = false
///   Architecture Performance Monitoring Features (0xa/edx):
///      number of fixed counters    = 0x0 (0)
///      bit width of fixed counters = 0x0 (0)
///      anythread deprecation       = false
///   x2APIC features / processor topology (0xb):
///      extended APIC ID                      = 10
///      --- level 0 ---
///      level number                          = 0x0 (0)
///      level type                            = thread (1)
///      bit width of level                    = 0x1 (1)
///      number of logical processors at level = 0x2 (2)
///      --- level 1 ---
///      level number                          = 0x1 (1)
///      level type                            = core (2)
///      bit width of level                    = 0x7 (7)
///      number of logical processors at level = 0xc (12)
///   XSAVE features (0xd/0):
///      XCR0 lower 32 bits valid bit field mask = 0x00000207
///      XCR0 upper 32 bits valid bit field mask = 0x00000000
///         XCR0 supported: x87 state            = true
///         XCR0 supported: SSE state            = true
///         XCR0 supported: AVX state            = true
///         XCR0 supported: MPX BNDREGS          = false
///         XCR0 supported: MPX BNDCSR           = false
///         XCR0 supported: AVX-512 opmask       = false
///         XCR0 supported: AVX-512 ZMM_Hi256    = false
///         XCR0 supported: AVX-512 Hi16_ZMM     = false
///         IA32_XSS supported: PT state         = false
///         XCR0 supported: PKRU state           = true
///         XCR0 supported: CET_U state          = false
///         XCR0 supported: CET_S state          = false
///         IA32_XSS supported: HDC state        = false
///      bytes required by fields in XCR0        = 0x00000340 (832)
///      bytes required by XSAVE/XRSTOR area     = 0x00000380 (896)
///   XSAVE features (0xd/1):
///      XSAVEOPT instruction                        = true
///      XSAVEC instruction                          = true
///      XGETBV instruction                          = true
///      XSAVES/XRSTORS instructions                 = true
///      SAVE area size in bytes                     = 0x00000340 (832)
///      IA32_XSS lower 32 bits valid bit field mask = 0x00000000
///      IA32_XSS upper 32 bits valid bit field mask = 0x00000000
///   AVX/YMM features (0xd/2):
///      AVX/YMM save state byte size             = 0x00000100 (256)
///      AVX/YMM save state byte offset           = 0x00000240 (576)
///      supported in IA32_XSS or XCR0            = XCR0 (user state)
///      64-byte alignment in compacted XSAVE     = false
///   PKRU features (0xd/9):
///      PKRU save state byte size                = 0x00000040 (64)
///      PKRU save state byte offset              = 0x00000340 (832)
///      supported in IA32_XSS or XCR0            = XCR0 (user state)
///      64-byte alignment in compacted XSAVE     = false
///   Quality of Service Monitoring Resource Type (0xf/0):
///      Maximum range of RMID = 255
///      supports L3 cache QoS monitoring = true
///   L3 Cache Quality of Service Monitoring (0xf/1):
///      Conversion factor from IA32_QM_CTR to bytes = 64
///      Maximum range of RMID                       = 255
///      supports L3 occupancy monitoring       = true
///      supports L3 total bandwidth monitoring = true
///      supports L3 local bandwidth monitoring = true
///   Resource Director Technology Allocation (0x10/0):
///      L3 cache allocation technology supported = true
///      L2 cache allocation technology supported = false
///      memory bandwidth allocation supported    = false
///   L3 Cache Allocation Technology (0x10/1):
///      length of capacity bit mask              = 0x10 (16)
///      Bit-granular map of isolation/contention = 0x00000000
///      infrequent updates of COS                = false
///      code and data prioritization supported   = true
///      highest COS number supported             = 0xf (15)
///   extended processor signature (0x80000001/eax):
///      family/generation = 0xf (15)
///      model           = 0x1 (1)
///      stepping id     = 0x0 (0)
///      extended family = 0x8 (8)
///      extended model  = 0x7 (7)
///      (family synth)  = 0x17 (23)
///      (model synth)   = 0x71 (113)
///      (simple synth)  = AMD Ryzen (Matisse B0) [Zen 2], 7nm
///   extended feature flags (0x80000001/edx):
///      x87 FPU on chip                       = true
///      virtual-8086 mode enhancement         = true
///      debugging extensions                  = true
///      page size extensions                  = true
///      time stamp counter                    = true
///      RDMSR and WRMSR support               = true
///      physical address extensions           = true
///      machine check exception               = true
///      CMPXCHG8B inst.                       = true
///      APIC on chip                          = true
///      SYSCALL and SYSRET instructions       = true
///      memory type range registers           = true
///      global paging extension               = true
///      machine check architecture            = true
///      conditional move/compare instruction  = true
///      page attribute table                  = true
///      page size extension                   = true
///      multiprocessing capable               = false
///      no-execute page protection            = true
///      AMD multimedia instruction extensions = true
///      MMX Technology                        = true
///      FXSAVE/FXRSTOR                        = true
///      SSE extensions                        = true
///      1-GB large page support               = true
///      RDTSCP                                = true
///      long mode (AA-64)                     = true
///      3DNow! instruction extensions         = false
///      3DNow! instructions                   = false
///   extended brand id (0x80000001/ebx):
///      raw     = 0x20000000 (536870912)
///      BrandId = 0x0 (0)
///      PkgType = AM4 (2)
///   AMD feature flags (0x80000001/ecx):
///      LAHF/SAHF supported in 64-bit mode     = true
///      CMP Legacy                             = true
///      SVM: secure virtual machine            = true
///      extended APIC space                    = true
///      AltMovCr8                              = true
///      LZCNT advanced bit manipulation        = true
///      SSE4A support                          = true
///      misaligned SSE mode                    = true
///      3DNow! PREFETCH/PREFETCHW instructions = true
///      OS visible workaround                  = true
///      instruction based sampling             = true
///      XOP support                            = false
///      SKINIT/STGI support                    = true
///      watchdog timer support                 = true
///      lightweight profiling support          = false
///      4-operand FMA instruction              = false
///      TCE: translation cache extension       = true
///      NodeId MSR C001100C                    = false
///      TBM support                            = false
///      topology extensions                    = true
///      core performance counter extensions    = true
///      NB/DF performance counter extensions   = true
///      data breakpoint extension              = true
///      performance time-stamp counter support = false
///      LLC performance counter extensions     = true
///      MWAITX/MONITORX supported              = true
///      Address mask extension support         = true
///   brand = "AMD Ryzen 5 3600X 6-Core Processor             "
///   L1 TLB/cache information: 2M/4M pages & L1 TLB (0x80000005/eax):
///      instruction # entries     = 0x40 (64)
///      instruction associativity = 0xff (255)
///      data # entries            = 0x40 (64)
///      data associativity        = 0xff (255)
///   L1 TLB/cache information: 4K pages & L1 TLB (0x80000005/ebx):
///      instruction # entries     = 0x40 (64)
///      instruction associativity = 0xff (255)
///      data # entries            = 0x40 (64)
///      data associativity        = 0xff (255)
///   L1 data cache information (0x80000005/ecx):
///      line size (bytes) = 0x40 (64)
///      lines per tag     = 0x1 (1)
///      associativity     = 0x8 (8)
///      size (KB)         = 0x20 (32)
///   L1 instruction cache information (0x80000005/edx):
///      line size (bytes) = 0x40 (64)
///      lines per tag     = 0x1 (1)
///      associativity     = 0x8 (8)
///      size (KB)         = 0x20 (32)
///   L2 TLB/cache information: 2M/4M pages & L2 TLB (0x80000006/eax):
///      instruction # entries     = 0x400 (1024)
///      instruction associativity = 8-way (6)
///      data # entries            = 0x800 (2048)
///      data associativity        = 4-way (4)
///   L2 TLB/cache information: 4K pages & L2 TLB (0x80000006/ebx):
///      instruction # entries     = 0x400 (1024)
///      instruction associativity = 8-way (6)
///      data # entries            = 0x800 (2048)
///      data associativity        = 8-way (6)
///   L2 unified cache information (0x80000006/ecx):
///      line size (bytes) = 0x40 (64)
///      lines per tag     = 0x1 (1)
///      associativity     = 8-way (6)
///      size (KB)         = 0x200 (512)
///   L3 cache information (0x80000006/edx):
///      line size (bytes)     = 0x40 (64)
///      lines per tag         = 0x1 (1)
///      associativity         = 0x9 (9)
///      size (in 512KB units) = 0x40 (64)
///   RAS Capability (0x80000007/ebx):
///      MCA overflow recovery support = true
///      SUCCOR support                = true
///      HWA: hardware assert support  = false
///      scalable MCA support          = true
///   Advanced Power Management Features (0x80000007/ecx):
///      CmpUnitPwrSampleTimeRatio = 0x0 (0)
///   Advanced Power Management Features (0x80000007/edx):
///      TS: temperature sensing diode           = true
///      FID: frequency ID control               = false
///      VID: voltage ID control                 = false
///      TTP: thermal trip                       = true
///      TM: thermal monitor                     = true
///      STC: software thermal control           = false
///      100 MHz multiplier control              = false
///      hardware P-State control                = true
///      TscInvariant                            = true
///      CPB: core performance boost             = true
///      read-only effective frequency interface = true
///      processor feedback interface            = false
///      APM power reporting                     = false
///      connected standby                       = true
///      RAPL: running average power limit       = true
///   Physical Address and Linear Address Size (0x80000008/eax):
///      maximum physical address bits         = 0x30 (48)
///      maximum linear (virtual) address bits = 0x30 (48)
///      maximum guest physical address bits   = 0x0 (0)
///   Extended Feature Extensions ID (0x80000008/ebx):
///      CLZERO instruction                       = true
///      instructions retired count support       = true
///      always save/restore error pointers       = true
///      RDPRU instruction                        = true
///      memory bandwidth enforcement             = true
///      WBNOINVD instruction                     = true
///      IBPB: indirect branch prediction barrier = true
///      IBRS: indirect branch restr speculation  = false
///      STIBP: 1 thr indirect branch predictor   = true
///      STIBP always on preferred mode           = true
///      ppin processor id number supported       = false
///      SSBD: speculative store bypass disable   = true
///      virtualized SSBD                         = false
///      SSBD fixed in hardware                   = false
///   Size Identifiers (0x80000008/ecx):
///      number of threads                   = 0xc (12)
///      ApicIdCoreIdSize                    = 0x7 (7)
///      performance time-stamp counter size = 0x0 (0)
///   Feature Extended Size (0x80000008/edx):
///      RDPRU instruction max input support = 0x1 (1)
///   SVM Secure Virtual Machine (0x8000000a/eax):
///      SvmRev: SVM revision = 0x1 (1)
///   SVM Secure Virtual Machine (0x8000000a/edx):
///      nested paging                           = true
///      LBR virtualization                      = true
///      SVM lock                                = true
///      NRIP save                               = true
///      MSR based TSC rate control              = true
///      VMCB clean bits support                 = true
///      flush by ASID                           = true
///      decode assists                          = true
///      SSSE3/SSE5 opcode set disable           = false
///      pause intercept filter                  = true
///      pause filter threshold                  = true
///      AVIC: AMD virtual interrupt controller  = true
///      virtualized VMLOAD/VMSAVE               = true
///      virtualized global interrupt flag (GIF) = true
///      GMET: guest mode execute trap           = true
///      guest Spec_ctl support                  = true
///   NASID: number of address space identifiers = 0x8000 (32768):
///   L1 TLB information: 1G pages (0x80000019/eax):
///      instruction # entries     = 0x40 (64)
///      instruction associativity = full (15)
///      data # entries            = 0x40 (64)
///      data associativity        = full (15)
///   L2 TLB information: 1G pages (0x80000019/ebx):
///      instruction # entries     = 0x0 (0)
///      instruction associativity = L2 off (0)
///      data # entries            = 0x0 (0)
///      data associativity        = L2 off (0)
///   SVM Secure Virtual Machine (0x8000001a/eax):
///      128-bit SSE executed full-width = false
///      MOVU* better than MOVL*/MOVH*   = true
///      256-bit SSE executed full-width = true
///   Instruction Based Sampling Identifiers (0x8000001b/eax):
///      IBS feature flags valid                  = true
///      IBS fetch sampling                       = true
///      IBS execution sampling                   = true
///      read write of op counter                 = true
///      op counting mode                         = true
///      branch target address reporting          = true
///      IbsOpCurCnt and IbsOpMaxCnt extend 7     = true
///      invalid RIP indication support           = true
///      fused branch micro-op indication support = true
///      IBS fetch control extended MSR support   = true
///      IBS op data 4 MSR support                = false
///   Lightweight Profiling Capabilities: Availability (0x8000001c/eax):
///      lightweight profiling                  = false
///      LWPVAL instruction                     = false
///      instruction retired event              = false
///      branch retired event                   = false
///      DC miss event                          = false
///      core clocks not halted event           = false
///      core reference clocks not halted event = false
///      interrupt on threshold overflow        = false
///   Lightweight Profiling Capabilities: Supported (0x8000001c/edx):
///      lightweight profiling                  = false
///      LWPVAL instruction                     = false
///      instruction retired event              = false
///      branch retired event                   = false
///      DC miss event                          = false
///      core clocks not halted event           = false
///      core reference clocks not halted event = false
///      interrupt on threshold overflow        = false
///   Lightweight Profiling Capabilities (0x8000001c/ebx):
///      LWPCB byte size             = 0x0 (0)
///      event record byte size      = 0x0 (0)
///      maximum EventId             = 0x0 (0)
///      EventInterval1 field offset = 0x0 (0)
///   Lightweight Profiling Capabilities (0x8000001c/ecx):
///      latency counter bit size          = 0x0 (0)
///      data cache miss address valid     = false
///      amount cache latency is rounded   = 0x0 (0)
///      LWP implementation version        = 0x0 (0)
///      event ring buffer size in records = 0x0 (0)
///      branch prediction filtering       = false
///      IP filtering                      = false
///      cache level filtering             = false
///      cache latency filteing            = false
///   Cache Properties (0x8000001d):
///      --- cache 0 ---
///      type                            = data (1)
///      level                           = 0x1 (1)
///      self-initializing               = true
///      fully associative               = false
///      extra cores sharing this cache  = 0x1 (1)
///      line size in bytes              = 0x40 (64)
///      physical line partitions        = 0x1 (1)
///      number of ways                  = 0x8 (8)
///      number of sets                  = 64
///      write-back invalidate           = false
///      cache inclusive of lower levels = false
///      (synth size)                    = 32768 (32 KB)
///      --- cache 1 ---
///      type                            = instruction (2)
///      level                           = 0x1 (1)
///      self-initializing               = true
///      fully associative               = false
///      extra cores sharing this cache  = 0x1 (1)
///      line size in bytes              = 0x40 (64)
///      physical line partitions        = 0x1 (1)
///      number of ways                  = 0x8 (8)
///      number of sets                  = 64
///      write-back invalidate           = false
///      cache inclusive of lower levels = false
///      (synth size)                    = 32768 (32 KB)
///      --- cache 2 ---
///      type                            = unified (3)
///      level                           = 0x2 (2)
///      self-initializing               = true
///      fully associative               = false
///      extra cores sharing this cache  = 0x1 (1)
///      line size in bytes              = 0x40 (64)
///      physical line partitions        = 0x1 (1)
///      number of ways                  = 0x8 (8)
///      number of sets                  = 1024
///      write-back invalidate           = false
///      cache inclusive of lower levels = true
///      (synth size)                    = 524288 (512 KB)
///      --- cache 3 ---
///      type                            = unified (3)
///      level                           = 0x3 (3)
///      self-initializing               = true
///      fully associative               = false
///      extra cores sharing this cache  = 0x5 (5)
///      line size in bytes              = 0x40 (64)
///      physical line partitions        = 0x1 (1)
///      number of ways                  = 0x10 (16)
///      number of sets                  = 16384
///      write-back invalidate           = true
///      cache inclusive of lower levels = false
///      (synth size)                    = 16777216 (16 MB)
///   extended APIC ID = 10
///   Core Identifiers (0x8000001e/ebx):
///      core ID          = 0x5 (5)
///      threads per core = 0x2 (2)
///   Node Identifiers (0x8000001e/ecx):
///      node ID             = 0x0 (0)
///      nodes per processor = 0x1 (1)
///   AMD Secure Encryption (0x8000001f):
///      SME: secure memory encryption support    = true
///      SEV: secure encrypted virtualize support = true
///      VM page flush MSR support                = true
///      SEV-ES: SEV encrypted state support      = true
///      encryption bit position in PTE           = 0x2f (47)
///      physical address space width reduction   = 0x5 (5)
///      number of SEV-enabled guests supported   = 0x1fd (509)
///      minimum SEV guest ASID                   = 0x1 (1)
///   PQoS Enforcement for Memory Bandwidth (0x80000020):
///      memory bandwidth enforcement support = true
///      capacity bitmask length              = 0xc (12)
///      number of classes of service         = 0xf (15)
///   (instruction supported synth):
///      CMPXCHG8B                = true
///      conditional move/compare = true
///      PREFETCH/PREFETCHW       = true
///   (multi-processing synth) = multi-core (c=12)
///   (multi-processing method) = AMD
///   (APIC widths synth): CORE_width=3 SMT_width=1
///   (APIC synth): PKG_ID=0 CORE_ID=5 SMT_ID=0
///   (uarch synth) = AMD Zen 2, 7nm
///   (synth) = AMD Ryzen (Matisse B0) [Zen 2], 7nm
/// ```
static CPUID_VALUE_MAP: phf::Map<u64, CpuIdResult> = phf_map! {
    0x00000000_00000000u64 => CpuIdResult { eax: 0x00000010, ebx: 0x68747541, ecx: 0x444d4163, edx: 0x69746e65 },
    0x00000001_00000000u64 => CpuIdResult { eax: 0x00870f10, ebx: 0x000c0800, ecx: 0x7ed8320b, edx: 0x178bfbff },
    0x00000002_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x00000003_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x00000005_00000000u64 => CpuIdResult { eax: 0x00000040, ebx: 0x00000040, ecx: 0x00000003, edx: 0x00000011 },
    0x00000006_00000000u64 => CpuIdResult { eax: 0x00000004, ebx: 0x00000000, ecx: 0x00000001, edx: 0x00000000 },
    0x00000007_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x219c91a9, ecx: 0x00400004, edx: 0x00000000 },
    0x00000007_00000001u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x00000008_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x00000009_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000a_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000b_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x00000002, ecx: 0x00000100, edx: 0x00000000 },
    0x0000000b_00000001u64 => CpuIdResult { eax: 0x00000007, ebx: 0x0000000c, ecx: 0x00000201, edx: 0x00000000 },
    0x0000000c_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000d_00000000u64 => CpuIdResult { eax: 0x00000207, ebx: 0x00000340, ecx: 0x00000380, edx: 0x00000000 },
    0x0000000d_00000001u64 => CpuIdResult { eax: 0x0000000f, ebx: 0x00000340, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000d_00000002u64 => CpuIdResult { eax: 0x00000100, ebx: 0x00000240, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000d_00000009u64 => CpuIdResult { eax: 0x00000040, ebx: 0x00000340, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000e_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x0000000f_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x000000ff, ecx: 0x00000000, edx: 0x00000002 },
    0x0000000f_00000001u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000040, ecx: 0x000000ff, edx: 0x00000007 },
    0x00000010_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000002, ecx: 0x00000000, edx: 0x00000000 },
    0x00000010_00000001u64 => CpuIdResult { eax: 0x0000000f, ebx: 0x00000000, ecx: 0x00000004, edx: 0x0000000f },
    0x20000000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000000_00000000u64 => CpuIdResult { eax: 0x80000020, ebx: 0x68747541, ecx: 0x444d4163, edx: 0x69746e65 },
    0x80000001_00000000u64 => CpuIdResult { eax: 0x00870f10, ebx: 0x20000000, ecx: 0x75c237ff, edx: 0x2fd3fbff },
    0x80000002_00000000u64 => CpuIdResult { eax: 0x20444d41, ebx: 0x657a7952, ecx: 0x2035206e, edx: 0x30303633 },
    0x80000003_00000000u64 => CpuIdResult { eax: 0x2d362058, ebx: 0x65726f43, ecx: 0x6f725020, edx: 0x73736563 },
    0x80000004_00000000u64 => CpuIdResult { eax: 0x2020726f, ebx: 0x20202020, ecx: 0x20202020, edx: 0x00202020 },
    0x80000005_00000000u64 => CpuIdResult { eax: 0xff40ff40, ebx: 0xff40ff40, ecx: 0x20080140, edx: 0x20080140 },
    0x80000006_00000000u64 => CpuIdResult { eax: 0x48006400, ebx: 0x68006400, ecx: 0x02006140, edx: 0x01009140 },
    0x80000007_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x0000001b, ecx: 0x00000000, edx: 0x00006799 },
    0x80000008_00000000u64 => CpuIdResult { eax: 0x00003030, ebx: 0x010eb757, ecx: 0x0000700b, edx: 0x00010000 },
    0x80000009_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000000a_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x00008000, ecx: 0x00000000, edx: 0x0013bcff },
    0x8000000b_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000000c_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000000d_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000000e_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000000f_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000010_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000011_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000012_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000013_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000014_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000015_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000016_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000017_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000018_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x80000019_00000000u64 => CpuIdResult { eax: 0xf040f040, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000001a_00000000u64 => CpuIdResult { eax: 0x00000006, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000001b_00000000u64 => CpuIdResult { eax: 0x000003ff, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000001c_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0x8000001d_00000000u64 => CpuIdResult { eax: 0x00004121, ebx: 0x01c0003f, ecx: 0x0000003f, edx: 0x00000000 },
    0x8000001d_00000001u64 => CpuIdResult { eax: 0x00004122, ebx: 0x01c0003f, ecx: 0x0000003f, edx: 0x00000000 },
    0x8000001d_00000002u64 => CpuIdResult { eax: 0x00004143, ebx: 0x01c0003f, ecx: 0x000003ff, edx: 0x00000002 },
    0x8000001d_00000003u64 => CpuIdResult { eax: 0x00014163, ebx: 0x03c0003f, ecx: 0x00003fff, edx: 0x00000001 },
    0x8000001e_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000100, ecx: 0x00000000, edx: 0x00000000 },
    0x8000001f_00000000u64 => CpuIdResult { eax: 0x0001000f, ebx: 0x0000016f, ecx: 0x000001fd, edx: 0x00000001 },
    0x80000020_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000002, ecx: 0x00000000, edx: 0x00000000 },
    0x80000020_00000001u64 => CpuIdResult { eax: 0x0000000b, ebx: 0x00000000, ecx: 0x00000000, edx: 0x0000000f },
    0x80860000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
    0xc0000000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000, edx: 0x00000000 },
};

fn cpuid_reader(eax: u32, ecx: u32) -> CpuIdResult {
    let key = (eax as u64) << u32::BITS | ecx as u64;
    CPUID_VALUE_MAP[&key]
}

/// Check that vendor is AuthenticAMD.
#[test]
fn vendor_check() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let v = cpuid.get_vendor_info().expect("Need to find vendor info");
    assert_eq!(v.as_str(), "AuthenticAMD");
}

/// Check feature info gives correct values for CPU
#[test]
fn version_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let f = cpuid.get_feature_info().expect("Need to find feature info");

    assert_eq!(f.base_family_id(), 0xf);
    assert_eq!(f.base_model_id(), 0x1);
    assert_eq!(f.stepping_id(), 0x0);
    assert_eq!(f.extended_family_id(), 0x8);
    assert_eq!(f.extended_model_id(), 0x7);
    assert_eq!(f.brand_index(), 0x0);
    assert_eq!(f.cflush_cache_line_size(), 0x8);
    assert_eq!(f.max_logical_processor_ids(), 0xc);

    assert!(f.has_fpu());
    assert!(f.has_vme());
    assert!(f.has_de());
    assert!(f.has_pse());
    assert!(f.has_tsc());
    assert!(f.has_msr());
    assert!(f.has_pae());
    assert!(f.has_mce());
    assert!(f.has_cmpxchg8b());
    assert!(f.has_apic());
    assert!(f.has_sysenter_sysexit());
    assert!(f.has_mtrr());
    assert!(f.has_pge());
    assert!(f.has_mca());
    assert!(f.has_cmov());
    assert!(f.has_pat());
    assert!(f.has_pse36());
    assert!(!f.has_psn());
    assert!(f.has_clflush());
    assert!(!f.has_ds());
    assert!(!f.has_acpi());
    assert!(f.has_mmx());
    assert!(f.has_fxsave_fxstor());
    assert!(f.has_sse());
    assert!(f.has_sse2());
    assert!(!f.has_ss());
    assert!(f.has_htt());
    assert!(!f.has_tm());
    assert!(!f.has_pbe());

    assert!(f.has_sse3());
    assert!(f.has_pclmulqdq());
    assert!(!f.has_ds_area());
    assert!(f.has_monitor_mwait());
    assert!(!f.has_cpl());
    assert!(!f.has_vmx());
    assert!(!f.has_smx());
    assert!(!f.has_eist());
    assert!(!f.has_tm2());
    assert!(f.has_ssse3());
    assert!(!f.has_cnxtid());
    // has_SDBG
    assert!(f.has_fma());
    assert!(f.has_cmpxchg16b());
    // xTPR
    assert!(!f.has_pdcm());
    assert!(!f.has_pcid());
    assert!(!f.has_dca());
    assert!(f.has_sse41());
    assert!(f.has_sse42());
    assert!(!f.has_x2apic());
    assert!(f.has_movbe());
    assert!(f.has_popcnt());
    assert!(!f.has_tsc_deadline());
    assert!(f.has_aesni());
    assert!(f.has_xsave());
    assert!(f.has_oxsave());
    assert!(f.has_avx());
    assert!(f.has_f16c());
    assert!(f.has_rdrand());
    assert!(!f.has_hypervisor());
}

#[test]
fn cache_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(cpuid.get_cache_info().is_none(), "Not supported by AMD");
}

#[test]
fn processor_serial() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(
        cpuid.get_processor_serial().is_none(),
        "Not supported by AMD"
    );
}

#[test]
fn monitor_mwait() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mw = cpuid.get_monitor_mwait_info().expect("Leaf is supported");
    assert_eq!(mw.largest_monitor_line(), 64);
    assert_eq!(mw.smallest_monitor_line(), 64);
    assert!(mw.interrupts_as_break_event());
    assert!(mw.extensions_supported());
    // supported_cX_states functions are not supported according to the manual
}

#[test]
fn thermal_power() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mw = cpuid.get_thermal_power_info().expect("Leaf is supported");

    assert_eq!(mw.dts_irq_threshold(), 0x0);
    assert!(!mw.has_dts());
    assert!(mw.has_arat());
    assert!(!mw.has_turbo_boost());
    assert!(!mw.has_pln());
    assert!(!mw.has_ecmd());
    assert!(!mw.has_ptm());
    assert!(!mw.has_hwp());
    assert!(!mw.has_hwp_notification());
    assert!(!mw.has_hwp_activity_window());
    assert!(!mw.has_hwp_energy_performance_preference());
    assert!(!mw.has_hwp_package_level_request());
    assert!(!mw.has_hdc());
    assert!(!mw.has_turbo_boost3());
    assert!(!mw.has_hwp_capabilities());
    assert!(!mw.has_hwp_peci_override());
    assert!(!mw.has_flexible_hwp());
    assert!(!mw.has_hwp_fast_access_mode());
    assert!(!mw.has_ignore_idle_processor_hwp_request());
    assert!(mw.has_hw_coord_feedback());
    assert!(!mw.has_energy_bias_pref());
}

#[test]
fn extended_features() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_extended_feature_info()
        .expect("Leaf is supported");

    assert!(e.has_fsgsbase());
    assert!(!e.has_tsc_adjust_msr());
    assert!(e.has_bmi1());
    assert!(!e.has_hle());
    assert!(e.has_avx2());
    assert!(!e.has_fdp());
    assert!(e.has_smep());
    assert!(e.has_bmi2());
    assert!(!e.has_rep_movsb_stosb());
    assert!(!e.has_invpcid());
    assert!(!e.has_rtm());
    assert!(e.has_rdtm());
    assert!(!e.has_fpu_cs_ds_deprecated());
    assert!(!e.has_mpx());
    assert!(e.has_rdta());
    assert!(e.has_rdseed());
    assert!(e.has_adx());
    assert!(e.has_smap());
    assert!(e.has_clflushopt());
    assert!(!e.has_processor_trace());
    assert!(e.has_sha());
    assert!(!e.has_sgx());
    assert!(!e.has_avx512f());
    assert!(!e.has_avx512dq());
    assert!(!e.has_avx512_ifma());
    assert!(!e.has_avx512pf());
    assert!(!e.has_avx512er());
    assert!(!e.has_avx512cd());
    assert!(!e.has_avx512bw());
    assert!(!e.has_avx512vl());
    assert!(e.has_clwb());
    assert!(!e.has_prefetchwt1());
    assert!(e.has_umip());
    assert!(!e.has_pku());
    assert!(!e.has_ospke());
    assert!(!e.has_avx512vnni());
    assert!(e.has_rdpid());
    assert!(!e.has_sgx_lc());
    assert_eq!(e.mawau_value(), 0x0);
}

#[test]
fn direct_cache_access() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(
        cpuid.get_direct_cache_access_info().is_none(),
        "Not supported by AMD"
    );
}

#[test]
fn perfmon_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(
        cpuid.get_performance_monitoring_info().is_none(),
        "Not supported by AMD"
    );
}

#[test]
fn extended_topology_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mut e = cpuid
        .get_extended_topology_info()
        .expect("Leaf is supported");

    let t = e.next().expect("Have level 0");
    assert_eq!(t.processors(), 2);
    assert_eq!(t.level_number(), 0);
    assert_eq!(t.level_type(), TopologyType::SMT);
    assert_eq!(t.x2apic_id(), 0x0);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x1);

    let t = e.next().expect("Have level 1");
    assert_eq!(t.processors(), 12);
    assert_eq!(t.level_number(), 1);
    assert_eq!(t.level_type(), TopologyType::Core);
    assert_eq!(t.x2apic_id(), 0x0);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x7);
}

#[test]
fn extended_state_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_extended_state_info().expect("Leaf is supported");

    assert!(e.xcr0_supports_legacy_x87());
    assert!(e.xcr0_supports_sse_128());
    assert!(e.xcr0_supports_avx_256());
    assert!(!e.xcr0_supports_mpx_bndregs());
    assert!(!e.xcr0_supports_mpx_bndcsr());
    assert!(!e.xcr0_supports_avx512_opmask());
    assert!(!e.xcr0_supports_avx512_zmm_hi256());
    assert!(!e.xcr0_supports_avx512_zmm_hi16());
    assert!(e.xcr0_supports_pkru());
    assert!(!e.ia32_xss_supports_pt());
    assert!(!e.ia32_xss_supports_hdc());
    assert_eq!(e.xsave_area_size_enabled_features(), 0x00000340);
    assert_eq!(e.xsave_area_size_supported_features(), 0x00000380);
    assert!(e.has_xsaveopt());
    assert!(e.has_xsavec());
    assert!(e.has_xgetbv());
    assert!(e.has_xsaves_xrstors());
    assert_eq!(e.xsave_size(), 0x00000340);

    let mut e = e.iter();
    let ee = e.next().expect("Has level 2");
    assert_eq!(ee.size(), 256);
    assert_eq!(ee.offset(), 576);
    assert!(ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 9");
    assert_eq!(ee.size(), 64);
    assert_eq!(ee.offset(), 832);
    assert!(ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());
}

#[test]
fn rdt_monitoring_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_rdt_monitoring_info().expect("Leaf is supported");

    assert!(e.has_l3_monitoring());
    assert_eq!(e.rmid_range(), 255);

    let l3m = e.l3_monitoring().expect("Leaf is available");
    assert_eq!(l3m.conversion_factor(), 64);
    assert_eq!(l3m.maximum_rmid_range(), 255);
    assert!(l3m.has_occupancy_monitoring());
    assert!(l3m.has_total_bandwidth_monitoring());
    assert!(l3m.has_local_bandwidth_monitoring());
}

#[test]
fn rdt_allocation_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_rdt_allocation_info().expect("Leaf is supported");

    assert!(e.has_l3_cat());
    assert!(!e.has_l2_cat());
    assert!(!e.has_memory_bandwidth_allocation());
    assert!(e.l2_cat().is_none());
    assert!(e.memory_bandwidth_allocation().is_none());

    let l3c = e.l3_cat().expect("Leaf is available");
    assert_eq!(l3c.capacity_mask_length(), 0x10);
    assert_eq!(l3c.isolation_bitmap(), 0x0);
    assert_eq!(l3c.highest_cos(), 15);
    assert!(l3c.has_code_data_prioritization());
}

#[test]
fn extended_processor_and_feature_identifiers() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_extended_processor_and_feature_identifiers()
        .expect("Leaf is supported");

    assert_eq!(e.pkg_type(), 0x2);
    assert_eq!(e.brand_id(), 0x0);

    assert!(e.has_lahf_sahf());
    assert!(e.has_cmp_legacy());
    assert!(e.has_svm());
    assert!(e.has_ext_apic_space());
    assert!(e.has_alt_mov_cr8());
    assert!(e.has_lzcnt());
    assert!(e.has_sse4a());
    assert!(e.has_misaligned_sse_mode());
    assert!(e.has_prefetchw());
    assert!(e.has_osvw());
    assert!(e.has_ibs());
    assert!(!e.has_xop());
    assert!(e.has_skinit());
    assert!(e.has_wdt());
    assert!(!e.has_lwp());
    assert!(!e.has_fma4());
    assert!(!e.has_tbm());
    assert!(e.has_topology_extensions());
    assert!(e.has_perf_cntr_extensions());
    assert!(e.has_nb_perf_cntr_extensions());
    assert!(e.has_data_access_bkpt_extension());
    assert!(!e.has_perf_tsc());
    assert!(e.has_perf_cntr_llc_extensions());
    assert!(e.has_monitorx_mwaitx());
    assert!(e.has_addr_mask_extension());
    assert!(e.has_syscall_sysret());
    assert!(e.has_execute_disable());
    assert!(e.has_mmx_extensions());
    assert!(e.has_fast_fxsave_fxstor());
    assert!(e.has_1gib_pages());
    assert!(e.has_rdtscp());
    assert!(e.has_64bit_mode());
    assert!(!e.has_amd_3dnow_extensions());
    assert!(!e.has_3dnow());
}

#[test]
fn brand_string() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_processor_brand_string()
        .expect("Leaf is supported");

    assert_eq!(e.as_str(), "AMD Ryzen 5 3600X 6-Core Processor");
}

#[test]
fn l1_tlb_cache() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_l1_cache_and_tlb_info()
        .expect("Leaf is supported");

    assert_eq!(
        e.dtlb_2m_4m_associativity(),
        Associativity::FullyAssociative
    );
    assert_eq!(e.dtlb_2m_4m_size(), 64);

    assert_eq!(
        e.itlb_2m_4m_associativity(),
        Associativity::FullyAssociative
    );
    assert_eq!(e.itlb_2m_4m_size(), 64);

    assert_eq!(e.dtlb_4k_associativity(), Associativity::FullyAssociative);
    assert_eq!(e.dtlb_4k_size(), 64);
    assert_eq!(e.itlb_4k_associativity(), Associativity::FullyAssociative);
    assert_eq!(e.itlb_4k_size(), 64);

    assert_eq!(e.dcache_line_size(), 64);
    assert_eq!(e.dcache_lines_per_tag(), 1);
    assert_eq!(e.dcache_associativity(), Associativity::NWay(8));
    assert_eq!(e.dcache_size(), 32);

    assert_eq!(e.icache_line_size(), 64);
    assert_eq!(e.icache_lines_per_tag(), 1);
    assert_eq!(e.icache_associativity(), Associativity::NWay(8));
    assert_eq!(e.icache_size(), 32);
}

#[test]
fn l2_l3_tlb_cache() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_l2_l3_cache_and_tlb_info()
        .expect("Leaf is supported");

    assert_eq!(e.itlb_2m_4m_associativity(), Associativity::NWay(8));
    assert_eq!(e.itlb_2m_4m_size(), 1024);

    assert_eq!(e.dtlb_2m_4m_associativity(), Associativity::NWay(4));
    assert_eq!(e.dtlb_2m_4m_size(), 2048);

    assert_eq!(e.itlb_4k_size(), 1024);
    assert_eq!(e.itlb_4k_associativity(), Associativity::NWay(8));

    assert_eq!(e.dtlb_4k_size(), 2048);
    assert_eq!(e.dtlb_4k_associativity(), Associativity::NWay(8));

    assert_eq!(e.l2cache_line_size(), 64);
    assert_eq!(e.l2cache_lines_per_tag(), 1);
    assert_eq!(e.l2cache_associativity(), Associativity::NWay(8));
    assert_eq!(e.l2cache_size(), 0x200);

    assert_eq!(e.l3cache_line_size(), 64);
    assert_eq!(e.l3cache_lines_per_tag(), 1);
    assert_eq!(e.l3cache_associativity(), Associativity::Unknown);
    assert_eq!(e.l3cache_size(), 64);
}

#[test]
fn apm() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_advanced_power_mgmt_info()
        .expect("Leaf is supported");

    assert_eq!(e.cpu_pwr_sample_time_ratio(), 0x0);

    assert!(e.has_mca_overflow_recovery());
    assert!(e.has_succor());
    assert!(!e.has_hwa());

    assert!(e.has_ts());
    assert!(!e.has_freq_id_ctrl());
    assert!(!e.has_volt_id_ctrl());
    assert!(e.has_thermtrip());
    assert!(e.has_tm());
    assert!(!e.has_100mhz_steps());
    assert!(e.has_hw_pstate());
    assert!(e.has_invariant_tsc());
    assert!(e.has_cpb());
    assert!(e.has_ro_effective_freq_iface());
    assert!(!e.has_feedback_iface());
    assert!(!e.has_power_reporting_iface());
}

#[test]
fn processor_capcity_features() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_processor_capacity_feature_info()
        .expect("Leaf is supported");

    assert_eq!(e.physical_address_bits(), 48);
    assert_eq!(e.linear_address_bits(), 48);
    assert_eq!(e.guest_physical_address_bits(), 0);

    // These are hard to test if they are correct. I think the cpuid CLI tool
    // displays bogus values here (see above) -- or I can't tell how they
    // correspond to the AMD manual...
    assert!(e.has_cl_zero());
    assert!(e.has_inst_ret_cntr_msr());
    assert!(e.has_restore_fp_error_ptrs());
    assert!(!e.has_invlpgb());
    assert!(e.has_rdpru());
    assert!(e.has_mcommit());
    assert!(e.has_wbnoinvd());
    assert!(e.has_int_wbinvd());
    assert!(!e.has_unsupported_efer_lmsle());
    assert!(!e.has_invlpgb_nested());

    assert_eq!(e.invlpgb_max_pages(), 0x0);
    assert_eq!(e.maximum_logical_processors(), 128);
    assert_eq!(e.num_phys_threads(), 12);
    assert_eq!(e.apic_id_size(), 7);
    assert_eq!(e.perf_tsc_size(), 40);
    assert_eq!(e.max_rdpru_id(), 0x1);
}

#[test]
fn secure_encryption() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_memory_encryption_info()
        .expect("Leaf is supported");

    assert!(e.has_sme());
    assert!(e.has_sev());
    assert!(e.has_page_flush_msr());
    assert!(e.has_sev_es());
    assert!(!e.has_sev_snp());
    assert!(!e.has_vmpl());
    assert!(!e.has_hw_enforced_cache_coh());
    assert!(!e.has_64bit_mode());
    assert!(!e.has_restricted_injection());
    assert!(!e.has_alternate_injection());
    assert!(!e.has_debug_swap());
    assert!(!e.has_prevent_host_ibs());
    assert!(e.has_vte());

    assert_eq!(e.c_bit_position(), 0x2f);
    assert_eq!(e.physical_address_reduction(), 0x5);
    assert_eq!(e.max_encrypted_guests(), 0x1fd);
    assert_eq!(e.min_sev_no_es_asid(), 0x1);
}

#[test]
fn svm() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_svm_info().expect("Leaf is supported");

    assert_eq!(e.revision(), 0x1);
    assert_eq!(e.supported_asids(), 0x8000);

    assert!(e.has_nested_paging());
    assert!(e.has_lbr_virtualization());
    assert!(e.has_svm_lock());
    assert!(e.has_nrip());
    assert!(e.has_tsc_rate_msr());
    assert!(e.has_vmcb_clean_bits());
    assert!(e.has_flush_by_asid());
    assert!(e.has_decode_assists());
    assert!(e.has_pause_filter());
    assert!(e.has_pause_filter_threshold());
    assert!(e.has_avic());
    assert!(e.has_vmsave_virtualization());
    assert!(e.has_gmet());
    assert!(!e.has_sss_check());
    assert!(e.has_spec_ctrl());
    assert!(!e.has_host_mce_override());
    assert!(!e.has_tlb_ctrl());
}

#[test]
fn tlb_1gb_page_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_tlb_1gb_page_info().expect("Leaf is supported");

    assert!(e.dtlb_l1_1gb_associativity() == Associativity::FullyAssociative);
    assert!(e.dtlb_l1_1gb_size() == 64);
    assert!(e.itlb_l1_1gb_associativity() == Associativity::FullyAssociative);
    assert!(e.itlb_l1_1gb_size() == 64);
    assert!(e.dtlb_l2_1gb_associativity() == Associativity::Disabled);
    assert!(e.dtlb_l2_1gb_size() == 0);
    assert!(e.itlb_l2_1gb_associativity() == Associativity::Disabled);
    assert!(e.itlb_l2_1gb_size() == 0);
}

#[test]
fn performance_optimization_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_performance_optimization_info()
        .expect("Leaf is supported");

    assert!(!e.has_fp128());
    assert!(e.has_movu());
    assert!(e.has_fp256());
}

#[test]
fn instruction_based_sampling_capabilities() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_instruction_based_sampling_capabilities()
        .expect("Leaf is supported");

    assert!(e.has_feature_flags());
    assert!(e.has_fetch_sampling());
    assert!(e.has_execution_sampling());
    assert!(e.has_read_write_operation_counter());
    assert!(e.has_operation_counter());
    assert!(e.has_branch_target_address_reporting());
    assert!(e.has_operation_counter_extended());
    assert!(e.has_invalid_rip_indication());
    assert!(e.has_fused_branch_micro_op_indication());
    assert!(!e.has_l3_miss_filtering());
}

#[test]
fn processor_topology_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_processor_topology_info()
        .expect("Leaf is supported");

    assert!(e.x2apic_id() == 0);
    assert!(e.core_id() == 0);
    assert!(e.threads_per_core() == 2);
    assert!(e.node_id() == 0);
    assert!(e.nodes_per_processor() == 1);
}

#[test]
fn remaining_unsupported_leafs() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);

    assert!(cpuid.get_sgx_info().is_none());
    assert!(cpuid.get_processor_trace_info().is_none());
    assert!(cpuid.get_tsc_info().is_none());
    assert!(cpuid.get_processor_frequency_info().is_none());
    assert!(cpuid.get_deterministic_address_translation_info().is_none());
    assert!(cpuid.get_soc_vendor_info().is_none());
    assert!(cpuid.get_extended_topology_info_v2().is_none());
    assert!(cpuid.get_extended_feature_identification_2().is_none());
    assert!(cpuid
        .get_extended_performance_monitoring_and_debug()
        .is_none());
    assert!(cpuid
        .get_multi_key_encrypted_memory_capabilities()
        .is_none());
    assert!(cpuid.get_extended_cpu_topology().is_none());
}

#[test]
fn platform_quality_service() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_pqos_extended_feature_info()
        .expect("Leaf is supported");

    assert!(e.has_l3mbe());
    assert!(!e.has_l3smbe());
    assert!(!e.has_abmc());
    assert!(!e.has_bmec());
    assert!(!e.has_l3rr());
    assert!(!e.has_sdciae());

    let f = e
        .get_l3_memory_bandwidth_enforcement_info()
        .expect("SubLeaf is supported");

    assert!(f.bandwidth_length() == 0x0000_000b);
    assert!(f.cos_max() == 0x0000_000f);
}
