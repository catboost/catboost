use crate::{CpuId, CpuIdResult};

// CPU:
//    vendor_id = "GenuineIntel"
//    version information (1/eax):
//       processor type  = primary processor (0)
//       family          = 0x6 (6)
//       model           = 0x7 (7)
//       stepping id     = 0x2 (2)
//       extended family = 0x0 (0)
//       extended model  = 0x9 (9)
//       (family synth)  = 0x6 (6)
//       (model synth)   = 0x97 (151)
//       (simple synth)  = Intel Atom (Alder Lake-S) [Golden Cove], 10nm
//    miscellaneous (1/ebx):
//       process local APIC physical ID = 0x0 (0)
//       maximum IDs for CPUs in pkg    = 0x80 (128)
//       CLFLUSH line size              = 0x8 (8)
//       brand index                    = 0x0 (0)
//    brand id = 0x00 (0): unknown
//    feature information (1/edx):
//       x87 FPU on chip                        = true
//       VME: virtual-8086 mode enhancement     = true
//       DE: debugging extensions               = true
//       PSE: page size extensions              = true
//       TSC: time stamp counter                = true
//       RDMSR and WRMSR support                = true
//       PAE: physical address extensions       = true
//       MCE: machine check exception           = true
//       CMPXCHG8B inst.                        = true
//       APIC on chip                           = true
//       SYSENTER and SYSEXIT                   = true
//       MTRR: memory type range registers      = true
//       PTE global bit                         = true
//       MCA: machine check architecture        = true
//       CMOV: conditional move/compare instr   = true
//       PAT: page attribute table              = true
//       PSE-36: page size extension            = true
//       PSN: processor serial number           = false
//       CLFLUSH instruction                    = true
//       DS: debug store                        = true
//       ACPI: thermal monitor and clock ctrl   = true
//       MMX Technology                         = true
//       FXSAVE/FXRSTOR                         = true
//       SSE extensions                         = true
//       SSE2 extensions                        = true
//       SS: self snoop                         = true
//       hyper-threading / multi-core supported = true
//       TM: therm. monitor                     = true
//       IA64                                   = false
//       PBE: pending break event               = true
//    feature information (1/ecx):
//       PNI/SSE3: Prescott New Instructions     = true
//       PCLMULDQ instruction                    = true
//       DTES64: 64-bit debug store              = true
//       MONITOR/MWAIT                           = true
//       CPL-qualified debug store               = true
//       VMX: virtual machine extensions         = true
//       SMX: safer mode extensions              = true
//       Enhanced Intel SpeedStep Technology     = true
//       TM2: thermal monitor 2                  = true
//       SSSE3 extensions                        = true
//       context ID: adaptive or shared L1 data  = false
//       SDBG: IA32_DEBUG_INTERFACE              = true
//       FMA instruction                         = true
//       CMPXCHG16B instruction                  = true
//       xTPR disable                            = true
//       PDCM: perfmon and debug                 = true
//       PCID: process context identifiers       = true
//       DCA: direct cache access                = false
//       SSE4.1 extensions                       = true
//       SSE4.2 extensions                       = true
//       x2APIC: extended xAPIC support          = true
//       MOVBE instruction                       = true
//       POPCNT instruction                      = true
//       time stamp counter deadline             = true
//       AES instruction                         = true
//       XSAVE/XSTOR states                      = true
//       OS-enabled XSAVE/XSTOR                  = true
//       AVX: advanced vector extensions         = true
//       F16C half-precision convert instruction = true
//       RDRAND instruction                      = true
//       hypervisor guest status                 = false
//    cache and TLB information (2):
//       0xff: cache data is in CPUID leaf 4
//       0xfe: TLB data is in CPUID leaf 0x18
//       0xf0: 64 byte prefetching
//    processor serial number = 0009-0672-0000-0000-0000-0000
//    deterministic cache parameters (4):
//       --- cache 0 ---
//       cache type                           = data cache (1)
//       cache level                          = 0x1 (1)
//       self-initializing cache level        = true
//       fully associative cache              = false
//       maximum IDs for CPUs sharing cache   = 0x1 (1)
//       maximum IDs for cores in pkg         = 0x3f (63)
//       system coherency line size           = 0x40 (64)
//       physical line partitions             = 0x1 (1)
//       ways of associativity                = 0xc (12)
//       number of sets                       = 0x40 (64)
//       WBINVD/INVD acts on lower caches     = false
//       inclusive to lower caches            = false
//       complex cache indexing               = false
//       number of sets (s)                   = 64
//       (size synth)                         = 49152 (48 KB)
//       --- cache 1 ---
//       cache type                           = instruction cache (2)
//       cache level                          = 0x1 (1)
//       self-initializing cache level        = true
//       fully associative cache              = false
//       maximum IDs for CPUs sharing cache   = 0x1 (1)
//       maximum IDs for cores in pkg         = 0x3f (63)
//       system coherency line size           = 0x40 (64)
//       physical line partitions             = 0x1 (1)
//       ways of associativity                = 0x8 (8)
//       number of sets                       = 0x40 (64)
//       WBINVD/INVD acts on lower caches     = false
//       inclusive to lower caches            = false
//       complex cache indexing               = false
//       number of sets (s)                   = 64
//       (size synth)                         = 32768 (32 KB)
//       --- cache 2 ---
//       cache type                           = unified cache (3)
//       cache level                          = 0x2 (2)
//       self-initializing cache level        = true
//       fully associative cache              = false
//       maximum IDs for CPUs sharing cache   = 0x7 (7)
//       maximum IDs for cores in pkg         = 0x3f (63)
//       system coherency line size           = 0x40 (64)
//       physical line partitions             = 0x1 (1)
//       ways of associativity                = 0xa (10)
//       number of sets                       = 0x800 (2048)
//       WBINVD/INVD acts on lower caches     = false
//       inclusive to lower caches            = false
//       complex cache indexing               = false
//       number of sets (s)                   = 2048
//       (size synth)                         = 1310720 (1.2 MB)
//       --- cache 3 ---
//       cache type                           = unified cache (3)
//       cache level                          = 0x3 (3)
//       self-initializing cache level        = true
//       fully associative cache              = false
//       maximum IDs for CPUs sharing cache   = 0x7f (127)
//       maximum IDs for cores in pkg         = 0x3f (63)
//       system coherency line size           = 0x40 (64)
//       physical line partitions             = 0x1 (1)
//       ways of associativity                = 0xa (10)
//       number of sets                       = 0xa000 (40960)
//       WBINVD/INVD acts on lower caches     = false
//       inclusive to lower caches            = false
//       complex cache indexing               = true
//       number of sets (s)                   = 40960
//       (size synth)                         = 26214400 (25 MB)
//    MONITOR/MWAIT (5):
//       smallest monitor-line size (bytes)       = 0x40 (64)
//       largest monitor-line size (bytes)        = 0x40 (64)
//       enum of Monitor-MWAIT exts supported     = true
//       supports intrs as break-event for MWAIT  = true
//       number of C0 sub C-states using MWAIT    = 0x0 (0)
//       number of C1 sub C-states using MWAIT    = 0x2 (2)
//       number of C2 sub C-states using MWAIT    = 0x0 (0)
//       number of C3 sub C-states using MWAIT    = 0x2 (2)
//       number of C4 sub C-states using MWAIT    = 0x0 (0)
//       number of C5 sub C-states using MWAIT    = 0x1 (1)
//       number of C6 sub C-states using MWAIT    = 0x0 (0)
//       number of C7 sub C-states using MWAIT    = 0x1 (1)
//    Thermal and Power Management Features (6):
//       digital thermometer                     = true
//       Intel Turbo Boost Technology            = true
//       ARAT always running APIC timer          = true
//       PLN power limit notification            = true
//       ECMD extended clock modulation duty     = true
//       PTM package thermal management          = true
//       HWP base registers                      = true
//       HWP notification                        = true
//       HWP activity window                     = true
//       HWP energy performance preference       = true
//       HWP package level request               = true
//       HDC base registers                      = false
//       Intel Turbo Boost Max Technology 3.0    = true
//       HWP capabilities                        = true
//       HWP PECI override                       = true
//       flexible HWP                            = true
//       IA32_HWP_REQUEST MSR fast access mode   = true
//       HW_FEEDBACK MSRs supported              = true
//       ignoring idle logical processor HWP req = true
//       enhanced hardware feedback interface    = true
//       digital thermometer thresholds          = 0x2 (2)
//       hardware coordination feedback          = true
//       ACNT2 available                         = false
//       performance-energy bias capability      = false
//       number of enh hardware feedback classes = 0x4 (4)
//       performance capability reporting        = true
//       energy efficiency capability reporting  = true
//       size of feedback struct (4KB pages)     = 0x1 (1)
//       index of CPU's row in feedback struct   = 0x0 (0)
//    extended feature flags (7):
//       FSGSBASE instructions                    = true
//       IA32_TSC_ADJUST MSR supported            = true
//       SGX: Software Guard Extensions supported = false
//       BMI1 instructions                        = true
//       HLE hardware lock elision                = false
//       AVX2: advanced vector extensions 2       = true
//       FDP_EXCPTN_ONLY                          = true
//       SMEP supervisor mode exec protection     = true
//       BMI2 instructions                        = true
//       enhanced REP MOVSB/STOSB                 = true
//       INVPCID instruction                      = true
//       RTM: restricted transactional memory     = false
//       RDT-CMT/PQoS cache monitoring            = false
//       deprecated FPU CS/DS                     = true
//       MPX: intel memory protection extensions  = false
//       RDT-CAT/PQE cache allocation             = false
//       AVX512F: AVX-512 foundation instructions = false
//       AVX512DQ: double & quadword instructions = false
//       RDSEED instruction                       = true
//       ADX instructions                         = true
//       SMAP: supervisor mode access prevention  = true
//       AVX512IFMA: fused multiply add           = false
//       PCOMMIT instruction                      = false
//       CLFLUSHOPT instruction                   = true
//       CLWB instruction                         = true
//       Intel processor trace                    = true
//       AVX512PF: prefetch instructions          = false
//       AVX512ER: exponent & reciprocal instrs   = false
//       AVX512CD: conflict detection instrs      = false
//       SHA instructions                         = true
//       AVX512BW: byte & word instructions       = false
//       AVX512VL: vector length                  = false
//       PREFETCHWT1                              = false
//       AVX512VBMI: vector byte manipulation     = false
//       UMIP: user-mode instruction prevention   = true
//       PKU protection keys for user-mode        = true
//       OSPKE CR4.PKE and RDPKRU/WRPKRU          = true
//       WAITPKG instructions                     = true
//       AVX512_VBMI2: byte VPCOMPRESS, VPEXPAND  = false
//       CET_SS: CET shadow stack                 = true
//       GFNI: Galois Field New Instructions      = true
//       VAES instructions                        = true
//       VPCLMULQDQ instruction                   = true
//       AVX512_VNNI: neural network instructions = false
//       AVX512_BITALG: bit count/shiffle         = false
//       TME: Total Memory Encryption             = true
//       AVX512: VPOPCNTDQ instruction            = false
//       5-level paging                           = false
//       BNDLDX/BNDSTX MAWAU value in 64-bit mode = 0x0 (0)
//       RDPID: read processor D supported        = true
//       KL: key locker                           = true
//       CLDEMOTE supports cache line demote      = false
//       MOVDIRI instruction                      = true
//       MOVDIR64B instruction                    = true
//       ENQCMD instruction                       = false
//       SGX_LC: SGX launch config supported      = false
//       PKS: supervisor protection keys          = true
//       AVX512_4VNNIW: neural network instrs     = false
//       AVX512_4FMAPS: multiply acc single prec  = false
//       fast short REP MOV                       = true
//       UINTR: user interrupts                   = false
//       AVX512_VP2INTERSECT: intersect mask regs = false
//       SRBDS mitigation MSR available           = false
//       VERW MD_CLEAR microcode support          = true
//       SERIALIZE instruction                    = true
//       hybrid part                              = true
//       TSXLDTRK: TSX suspend load addr tracking = false
//       PCONFIG instruction                      = true
//       LBR: architectural last branch records   = true
//       CET_IBT: CET indirect branch tracking    = true
//       AMX-BF16: tile bfloat16 support          = false
//       AVX512_FP16: fp16 support                = false
//       AMX-TILE: tile architecture support      = false
//       AMX-INT8: tile 8-bit integer support     = false
//       IBRS/IBPB: indirect branch restrictions  = true
//       STIBP: 1 thr indirect branch predictor   = true
//       L1D_FLUSH: IA32_FLUSH_CMD MSR            = true
//       IA32_ARCH_CAPABILITIES MSR               = true
//       IA32_CORE_CAPABILITIES MSR               = true
//       SSBD: speculative store bypass disable   = true
//       AVX-VNNI: AVX VNNI neural network instrs = true
//       AVX512_BF16: bfloat16 instructions       = false
//       zero-length MOVSB                        = false
//       fast short STOSB                         = true
//       fast short CMPSB, SCASB                  = false
//       HRESET: history reset support            = true
//    Direct Cache Access Parameters (9):
//    PLATFORM_DCA_CAP MSR bits = 0
//    Architecture Performance Monitoring Features (0xa):
//       version ID                               = 0x5 (5)
//       number of counters per logical processor = 0x6 (6)
//       bit width of counter                     = 0x30 (48)
//       length of EBX bit vector                 = 0x7 (7)
//       core cycle event not available           = false
//       instruction retired event not available  = false
//       reference cycles event not available     = false
//       last-level cache ref event not available = false
//       last-level cache miss event not avail    = false
//       branch inst retired event not available  = false
//       branch mispred retired event not avail   = false
//       fixed counter  0 supported               = true
//       fixed counter  1 supported               = true
//       fixed counter  2 supported               = true
//       fixed counter  3 supported               = false
//       fixed counter  4 supported               = false
//       fixed counter  5 supported               = false
//       fixed counter  6 supported               = false
//       fixed counter  7 supported               = false
//       fixed counter  8 supported               = false
//       fixed counter  9 supported               = false
//       fixed counter 10 supported               = false
//       fixed counter 11 supported               = false
//       fixed counter 12 supported               = false
//       fixed counter 13 supported               = false
//       fixed counter 14 supported               = false
//       fixed counter 15 supported               = false
//       fixed counter 16 supported               = false
//       fixed counter 17 supported               = false
//       fixed counter 18 supported               = false
//       fixed counter 19 supported               = false
//       fixed counter 20 supported               = false
//       fixed counter 21 supported               = false
//       fixed counter 22 supported               = false
//       fixed counter 23 supported               = false
//       fixed counter 24 supported               = false
//       fixed counter 25 supported               = false
//       fixed counter 26 supported               = false
//       fixed counter 27 supported               = false
//       fixed counter 28 supported               = false
//       fixed counter 29 supported               = false
//       fixed counter 30 supported               = false
//       fixed counter 31 supported               = false
//       number of fixed counters                 = 0x3 (3)
//       bit width of fixed counters              = 0x30 (48)
//       anythread deprecation                    = true
//    x2APIC features / processor topology (0xb):
//       extended APIC ID                      = 0
//       --- level 0 ---
//       level number                          = 0x0 (0)
//       level type                            = thread (1)
//       bit width of level                    = 0x1 (1)
//       number of logical processors at level = 0x2 (2)
//       --- level 1 ---
//       level number                          = 0x1 (1)
//       level type                            = core (2)
//       bit width of level                    = 0x7 (7)
//       number of logical processors at level = 0x14 (20)
//    XSAVE features (0xd/0):
//       XCR0 lower 32 bits valid bit field mask = 0x00000207
//       XCR0 upper 32 bits valid bit field mask = 0x00000000
//          XCR0 supported: x87 state            = true
//          XCR0 supported: SSE state            = true
//          XCR0 supported: AVX state            = true
//          XCR0 supported: MPX BNDREGS          = false
//          XCR0 supported: MPX BNDCSR           = false
//          XCR0 supported: AVX-512 opmask       = false
//          XCR0 supported: AVX-512 ZMM_Hi256    = false
//          XCR0 supported: AVX-512 Hi16_ZMM     = false
//          IA32_XSS supported: PT state         = false
//          XCR0 supported: PKRU state           = true
//          XCR0 supported: CET_U state          = false
//          XCR0 supported: CET_S state          = false
//          IA32_XSS supported: HDC state        = false
//          IA32_XSS supported: UINTR state      = false
//          LBR supported                        = false
//          IA32_XSS supported: HWP state        = false
//          XTILECFG supported                   = false
//          XTILEDATA supported                  = false
//       bytes required by fields in XCR0        = 0x00000a88 (2696)
//       bytes required by XSAVE/XRSTOR area     = 0x00000a88 (2696)
//    XSAVE features (0xd/1):
//       XSAVEOPT instruction                        = true
//       XSAVEC instruction                          = true
//       XGETBV instruction                          = true
//       XSAVES/XRSTORS instructions                 = true
//       XFD: extended feature disable supported     = false
//       SAVE area size in bytes                     = 0x00000670 (1648)
//       IA32_XSS lower 32 bits valid bit field mask = 0x00019900
//       IA32_XSS upper 32 bits valid bit field mask = 0x00000000
//    AVX/YMM features (0xd/2):
//       AVX/YMM save state byte size             = 0x00000100 (256)
//       AVX/YMM save state byte offset           = 0x00000240 (576)
//       supported in IA32_XSS or XCR0            = XCR0 (user state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    PT features (0xd/8):
//       PT save state byte size                  = 0x00000080 (128)
//       PT save state byte offset                = 0x00000000 (0)
//       supported in IA32_XSS or XCR0            = IA32_XSS (supervisor state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    PKRU features (0xd/9):
//       PKRU save state byte size                = 0x00000008 (8)
//       PKRU save state byte offset              = 0x00000a80 (2688)
//       supported in IA32_XSS or XCR0            = XCR0 (user state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    CET_U state features (0xd/0xb):
//       CET_U state save state byte size         = 0x00000010 (16)
//       CET_U state save state byte offset       = 0x00000000 (0)
//       supported in IA32_XSS or XCR0            = IA32_XSS (supervisor state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    CET_S state features (0xd/0xc):
//       CET_S state save state byte size         = 0x00000018 (24)
//       CET_S state save state byte offset       = 0x00000000 (0)
//       supported in IA32_XSS or XCR0            = IA32_XSS (supervisor state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    LBR features (0xd/0xf):
//       LBR save state byte size                 = 0x00000328 (808)
//       LBR save state byte offset               = 0x00000000 (0)
//       supported in IA32_XSS or XCR0            = IA32_XSS (supervisor state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    HWP state features (0xd/0x10):
//       HWP state save state byte size           = 0x00000008 (8)
//       HWP state save state byte offset         = 0x00000000 (0)
//       supported in IA32_XSS or XCR0            = IA32_XSS (supervisor state)
//       64-byte alignment in compacted XSAVE     = false
//       XFD faulting supported                   = false
//    Quality of Service Monitoring Resource Type (0xf/0):
//       Maximum range of RMID = 0
//       supports L3 cache QoS monitoring = false
//    Resource Director Technology Allocation (0x10/0):
//       L3 cache allocation technology supported = false
//       L2 cache allocation technology supported = false
//       memory bandwidth allocation supported    = false
//    0x00000011 0x00: eax=0x00000000 ebx=0x00000000 ecx=0x00000000 edx=0x00000000
//    Software Guard Extensions (SGX) capability (0x12/0):
//       SGX1 supported                         = false
//       SGX2 supported                         = false
//       SGX ENCLV E*VIRTCHILD, ESETCONTEXT     = false
//       SGX ENCLS ETRACKC, ERDINFO, ELDBC, ELDUC = false
//       MISCSELECT.EXINFO supported: #PF & #GP = false
//       MISCSELECT.CPINFO supported: #CP       = false
//       MaxEnclaveSize_Not64 (log2)            = 0x0 (0)
//       MaxEnclaveSize_64 (log2)               = 0x0 (0)
//    0x00000013 0x00: eax=0x00000000 ebx=0x00000000 ecx=0x00000000 edx=0x00000000
//    Intel Processor Trace (0x14):
//       IA32_RTIT_CR3_MATCH is accessible      = true
//       configurable PSB & cycle-accurate      = true
//       IP & TraceStop filtering; PT preserve  = true
//       MTC timing packet; suppress COFI-based = true
//       PTWRITE support                        = true
//       power event trace support              = false
//       ToPA output scheme support             = true
//       ToPA can hold many output entries      = true
//       single-range output scheme support     = true
//       output to trace transport              = false
//       IP payloads have LIP values & CS       = false
//       configurable address ranges            = 0x2 (2)
//       supported MTC periods bitmask          = 0x249 (585)
//       supported cycle threshold bitmask      = 0x3f (63)
//       supported config PSB freq bitmask      = 0x3f (63)
//       Time Stamp Counter/Core Crystal Clock Information (0x15):
//       TSC/clock ratio = 188/2
//       nominal core crystal clock = 38400000 Hz
//    Processor Frequency Information (0x16):
//       Core Base Frequency (MHz) = 0xe10 (3600)
//       Core Maximum Frequency (MHz) = 0x1388 (5000)
//       Bus (Reference) Frequency (MHz) = 0x64 (100)
//    System-On-Chip Vendor Attribute (0x17/0):
//       vendor id     = 0x0 (0)
//       vendor scheme = assigned by intel
//       project id  = 0x00000000 (0)
//       stepping id = 0x00000000 (0)
//    Deterministic Address Translation Parameters (0x18/0):
//       4KB page size entries supported = false
//       2MB page size entries supported = false
//       4MB page size entries supported = false
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x0 (0)
//       number of sets = 0x00000000 (0)
//       translation cache type            = invalid (0)
//       translation cache level           = 0x1 (1)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x0 (0)
//    Deterministic Address Translation Parameters (0x18/1):
//       4KB page size entries supported = true
//       2MB page size entries supported = false
//       4MB page size entries supported = false
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x8 (8)
//       number of sets = 0x00000020 (32)
//       translation cache type            = instruction TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/2):
//       4KB page size entries supported = false
//       2MB page size entries supported = true
//       4MB page size entries supported = true
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x8 (8)
//       number of sets = 0x00000004 (4)
//       translation cache type            = instruction TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/3):
//       4KB page size entries supported = true
//       2MB page size entries supported = true
//       4MB page size entries supported = true
//       1GB page size entries supported = true
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x10 (16)
//       number of sets = 0x00000001 (1)
//       translation cache type            = store-only TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = true
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/4):
//       4KB page size entries supported = true
//       2MB page size entries supported = false
//       4MB page size entries supported = false
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x4 (4)
//       number of sets = 0x00000010 (16)
//       translation cache type            = load-only TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/5):
//       4KB page size entries supported = false
//       2MB page size entries supported = true
//       4MB page size entries supported = true
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x4 (4)
//       number of sets = 0x00000008 (8)
//       translation cache type            = load-only TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/6):
//       4KB page size entries supported = false
//       2MB page size entries supported = false
//       4MB page size entries supported = false
//       1GB page size entries supported = true
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x8 (8)
//       number of sets = 0x00000001 (1)
//       translation cache type            = load-only TLB
//       translation cache level           = 0x2 (2)
//       fully associative                 = true
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/7):
//       4KB page size entries supported = true
//       2MB page size entries supported = true
//       4MB page size entries supported = true
//       1GB page size entries supported = false
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x8 (8)
//       number of sets = 0x00000080 (128)
//       translation cache type            = unified TLB
//       translation cache level           = 0x3 (3)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Deterministic Address Translation Parameters (0x18/8):
//       4KB page size entries supported = true
//       2MB page size entries supported = false
//       4MB page size entries supported = false
//       1GB page size entries supported = true
//       partitioning                    = soft between logical processors
//       ways of associativity           = 0x8 (8)
//       number of sets = 0x00000080 (128)
//       translation cache type            = unified TLB
//       translation cache level           = 0x3 (3)
//       fully associative                 = false
//       maximum number of addressible IDs = 0x1 (1)
//    Key Locker information (0x19):
//       CPL0-only restriction supported  = true
//       no-encrypt restriction supported = true
//       no-decrypt restriction supported = true
//       AESKLE: AES instructions         = false
//       AES wide instructions            = true
//       MSRs & IWKEY backups             = true
//       LOADIWKEY NoBackup parameter     = true
//       IWKEY randomization supported    = true
//    Hybrid Information (0x1a/0):
//       native model ID of core = 0x1 (1)
//       core type               = Intel Core
//    PCONFIG information (0x1b/n):
//       sub-leaf type = target identifier (1)
//       identifier of target 1 = 0x00000001 (1)
//       identifier of target 2 = 0x00000000 (0)
//       identifier of target 3 = 0x00000000 (0)
//    Architectural LBR Capabilities (0x1c/0):
//       IA32_LBR_DEPTH.DEPTH  8 supported = true
//       IA32_LBR_DEPTH.DEPTH 16 supported = true
//       IA32_LBR_DEPTH.DEPTH 24 supported = false
//       IA32_LBR_DEPTH.DEPTH 32 supported = true
//       IA32_LBR_DEPTH.DEPTH 40 supported = false
//       IA32_LBR_DEPTH.DEPTH 48 supported = false
//       IA32_LBR_DEPTH.DEPTH 56 supported = false
//       IA32_LBR_DEPTH.DEPTH 64 supported = false
//       deep C-state reset supported      = true
//       LBR IP values contain             = EIP (0)
//       CPL filtering supported           = true
//       branch filtering supported        = true
//       call-stack mode supported         = true
//       mispredict bit supported          = true
//       timed LBRs supported              = true
//       branch type field supported       = true
//    Tile Information (0x1d/0):
//       max_palette = 0
//    TMUL Information (0x1e/0):
//       tmul_maxk = 0x0 (0)
//       tmul_maxn = 0x0 (0)
//    V2 extended topology (0x1f):
//       x2APIC ID of logical processor = 0x0 (0)
//       --- level 0 ---
//       level number                          = 0x0 (0)
//       level type                            = thread (1)
//       bit width of level                    = 0x1 (1)
//       number of logical processors at level = 0x2 (2)
//       --- level 1 ---
//       level number                          = 0x1 (1)
//       level type                            = core (2)
//       bit width of level                    = 0x7 (7)
//       number of logical processors at level = 0x14 (20)
//       --- level 2 ---
//       level number                          = 0x2 (2)
//       level type                            = invalid (0)
//       bit width of level                    = 0x0 (0)
//       number of logical processors at level = 0x0 (0)
//    Processor History Reset information (0x20):
//       HRESET supported: EHFI history = true
//    extended feature flags (0x80000001/edx):
//       SYSCALL and SYSRET instructions        = true
//       execution disable                      = true
//       1-GB large page support                = true
//       RDTSCP                                 = true
//       64-bit extensions technology available = true
//    Intel feature flags (0x80000001/ecx):
//       LAHF/SAHF supported in 64-bit mode     = true
//       LZCNT advanced bit manipulation        = true
//       3DNow! PREFETCH/PREFETCHW instructions = true
//    brand = "12th Gen Intel(R) Core(TM) i7-12700K"
//    L1 TLB/cache information: 2M/4M pages & L1 TLB (0x80000005/eax):
//       instruction # entries     = 0x0 (0)
//       instruction associativity = 0x0 (0)
//       data # entries            = 0x0 (0)
//       data associativity        = 0x0 (0)
//    L1 TLB/cache information: 4K pages & L1 TLB (0x80000005/ebx):
//       instruction # entries     = 0x0 (0)
//       instruction associativity = 0x0 (0)
//       data # entries            = 0x0 (0)
//       data associativity        = 0x0 (0)
//    L1 data cache information (0x80000005/ecx):
//       line size (bytes) = 0x0 (0)
//       lines per tag     = 0x0 (0)
//       associativity     = 0x0 (0)
//       size (KB)         = 0x0 (0)
//    L1 instruction cache information (0x80000005/edx):
//       line size (bytes) = 0x0 (0)
//       lines per tag     = 0x0 (0)
//       associativity     = 0x0 (0)
//       size (KB)         = 0x0 (0)
//    L2 TLB/cache information: 2M/4M pages & L2 TLB (0x80000006/eax):
//       instruction # entries     = 0x0 (0)
//       instruction associativity = L2 off (0)
//       data # entries            = 0x0 (0)
//       data associativity        = L2 off (0)
//    L2 TLB/cache information: 4K pages & L2 TLB (0x80000006/ebx):
//       instruction # entries     = 0x0 (0)
//       instruction associativity = L2 off (0)
//       data # entries            = 0x0 (0)
//       data associativity        = L2 off (0)
//    L2 unified cache information (0x80000006/ecx):
//       line size (bytes) = 0x40 (64)
//       lines per tag     = 0x0 (0)
//       associativity     = 0x7 (7)
//       size (KB)         = 0x500 (1280)
//       L3 cache information (0x80000006/edx):
//       line size (bytes)     = 0x0 (0)
//       lines per tag         = 0x0 (0)
//       associativity         = L2 off (0)
//       size (in 512KB units) = 0x0 (0)
//    RAS Capability (0x80000007/ebx):
//       MCA overflow recovery support = false
//       SUCCOR support                = false
//       HWA: hardware assert support  = false
//       scalable MCA support          = false
//    Advanced Power Management Features (0x80000007/ecx):
//       CmpUnitPwrSampleTimeRatio = 0x0 (0)
//    Advanced Power Management Features (0x80000007/edx):
//       TS: temperature sensing diode           = false
//       FID: frequency ID control               = false
//       VID: voltage ID control                 = false
//       TTP: thermal trip                       = false
//       TM: thermal monitor                     = false
//       STC: software thermal control           = false
//       100 MHz multiplier control              = false
//       hardware P-State control                = false
//       TscInvariant                            = true
//       CPB: core performance boost             = false
//       read-only effective frequency interface = false
//       processor feedback interface            = false
//       APM power reporting                     = false
//       connected standby                       = false
//       RAPL: running average power limit       = false
//    Physical Address and Linear Address Size (0x80000008/eax):
//       maximum physical address bits         = 0x2e (46)
//       maximum linear (virtual) address bits = 0x30 (48)
//       maximum guest physical address bits   = 0x0 (0)
//    Extended Feature Extensions ID (0x80000008/ebx):
//       CLZERO instruction                       = false
//       instructions retired count support       = false
//       always save/restore error pointers       = false
//       RDPRU instruction                        = false
//       memory bandwidth enforcement             = false
//       WBNOINVD instruction                     = false
//       IBPB: indirect branch prediction barrier = false
//       IBRS: indirect branch restr speculation  = false
//       STIBP: 1 thr indirect branch predictor   = false
//       STIBP always on preferred mode           = false
//       ppin processor id number supported       = false
//       SSBD: speculative store bypass disable   = false
//       virtualized SSBD                         = false
//       SSBD fixed in hardware                   = false
//    Size Identifiers (0x80000008/ecx):
//       number of CPU cores                 = 0x1 (1)
//       ApicIdCoreIdSize                    = 0x0 (0)
//       performance time-stamp counter size = 0x0 (0)
//    Feature Extended Size (0x80000008/edx):
//       RDPRU instruction max input support = 0x0 (0)
//    (multi-processing synth) = multi-core (c=20), hyper-threaded (t=2)
//    (multi-processing method) = Intel leaf 0x1f
//    (APIC widths synth): CORE_width=7 SMT_width=1
//    (APIC synth): PKG_ID=0 CORE_ID=0 SMT_ID=0
//    (uarch synth) = Intel Golden Cove, 10nm
//    (synth) = Intel Atom (Alder Lake-S) [Golden Cove], 10nm

static CPUID_VALUE_MAP: phf::Map<u64, CpuIdResult> = phf::phf_map! {
    0x00000000_00000000u64 => CpuIdResult { eax: 0x00000020, ebx: 0x756e6547, ecx: 0x6c65746e,  edx: 0x49656e69 },
    0x00000001_00000000u64 => CpuIdResult { eax: 0x00090672, ebx: 0x00800800, ecx: 0x7ffafbff,  edx: 0xbfebfbff },
    0x00000002_00000000u64 => CpuIdResult { eax: 0x00feff01, ebx: 0x000000f0, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000003_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000004_00000000u64 => CpuIdResult { eax: 0xfc004121, ebx: 0x02c0003f, ecx: 0x0000003f,  edx: 0x00000000 },
    0x00000004_00000001u64 => CpuIdResult { eax: 0xfc004122, ebx: 0x01c0003f, ecx: 0x0000003f,  edx: 0x00000000 },
    0x00000004_00000002u64 => CpuIdResult { eax: 0xfc01c143, ebx: 0x0240003f, ecx: 0x000007ff,  edx: 0x00000000 },
    0x00000004_00000003u64 => CpuIdResult { eax: 0xfc1fc163, ebx: 0x0240003f, ecx: 0x00009fff,  edx: 0x00000004 },
    0x00000005_00000000u64 => CpuIdResult { eax: 0x00000040, ebx: 0x00000040, ecx: 0x00000003,  edx: 0x10102020 },
    0x00000006_00000000u64 => CpuIdResult { eax: 0x00dfcff7, ebx: 0x00000002, ecx: 0x00000401,  edx: 0x00000003 },
    0x00000007_00000000u64 => CpuIdResult { eax: 0x00000002, ebx: 0x239c27eb, ecx: 0x98c027bc,  edx: 0xfc1cc410 },
    0x00000007_00000001u64 => CpuIdResult { eax: 0x00400810, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000007_00000002u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000001 },
    0x00000008_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000009_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000000a_00000000u64 => CpuIdResult { eax: 0x07300605, ebx: 0x00000000, ecx: 0x00000007,  edx: 0x00008603 },
    0x0000000b_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x00000002, ecx: 0x00000100,  edx: 0x00000000 },
    0x0000000b_00000001u64 => CpuIdResult { eax: 0x00000007, ebx: 0x00000014, ecx: 0x00000201,  edx: 0x00000000 },
    0x0000000c_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000000d_00000000u64 => CpuIdResult { eax: 0x00000207, ebx: 0x00000a88, ecx: 0x00000a88,  edx: 0x00000000 },
    0x0000000d_00000001u64 => CpuIdResult { eax: 0x0000000f, ebx: 0x00000670, ecx: 0x00019900,  edx: 0x00000000 },
    0x0000000d_00000002u64 => CpuIdResult { eax: 0x00000100, ebx: 0x00000240, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000000d_00000008u64 => CpuIdResult { eax: 0x00000080, ebx: 0x00000000, ecx: 0x00000001,  edx: 0x00000000 },
    0x0000000d_00000009u64 => CpuIdResult { eax: 0x00000008, ebx: 0x00000a80, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000000d_0000000bu64 => CpuIdResult { eax: 0x00000010, ebx: 0x00000000, ecx: 0x00000001,  edx: 0x00000000 },
    0x0000000d_0000000cu64 => CpuIdResult { eax: 0x00000018, ebx: 0x00000000, ecx: 0x00000001,  edx: 0x00000000 },
    0x0000000d_0000000fu64 => CpuIdResult { eax: 0x00000328, ebx: 0x00000000, ecx: 0x00000001,  edx: 0x00000000 },
    0x0000000d_00000010u64 => CpuIdResult { eax: 0x00000008, ebx: 0x00000000, ecx: 0x00000001,  edx: 0x00000000 },
    0x0000000e_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000000f_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000010_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000011_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000012_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000013_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000014_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x0000005f, ecx: 0x00000007,  edx: 0x00000000 },
    0x00000014_00000001u64 => CpuIdResult { eax: 0x02490002, ebx: 0x003f003f, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000015_00000000u64 => CpuIdResult { eax: 0x00000002, ebx: 0x000000bc, ecx: 0x0249f000,  edx: 0x00000000 },
    0x00000016_00000000u64 => CpuIdResult { eax: 0x00000e10, ebx: 0x00001388, ecx: 0x00000064,  edx: 0x00000000 },
    0x00000017_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000018_00000000u64 => CpuIdResult { eax: 0x00000008, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x00000018_00000001u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00080001, ecx: 0x00000020,  edx: 0x00004022 },
    0x00000018_00000002u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00080006, ecx: 0x00000004,  edx: 0x00004022 },
    0x00000018_00000003u64 => CpuIdResult { eax: 0x00000000, ebx: 0x0010000f, ecx: 0x00000001,  edx: 0x00004125 },
    0x00000018_00000004u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00040001, ecx: 0x00000010,  edx: 0x00004024 },
    0x00000018_00000005u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00040006, ecx: 0x00000008,  edx: 0x00004024 },
    0x00000018_00000006u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00080008, ecx: 0x00000001,  edx: 0x00004124 },
    0x00000018_00000007u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00080007, ecx: 0x00000080,  edx: 0x00004043 },
    0x00000018_00000008u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00080009, ecx: 0x00000080,  edx: 0x00004043 },
    0x00000019_00000000u64 => CpuIdResult { eax: 0x00000007, ebx: 0x00000014, ecx: 0x00000003,  edx: 0x00000000 },
    0x0000001a_00000000u64 => CpuIdResult { eax: 0x40000001, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000001b_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x00000001, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000001c_00000000u64 => CpuIdResult { eax: 0x4000000b, ebx: 0x00000007, ecx: 0x00000007,  edx: 0x00000000 },
    0x0000001d_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000001e_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x0000001f_00000000u64 => CpuIdResult { eax: 0x00000001, ebx: 0x00000002, ecx: 0x00000100,  edx: 0x00000000 },
    0x0000001f_00000001u64 => CpuIdResult { eax: 0x00000007, ebx: 0x00000014, ecx: 0x00000201,  edx: 0x00000000 },
    0x0000001f_00000002u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000002,  edx: 0x00000000 },
    0x00000020_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000001, ecx: 0x00000000,  edx: 0x00000000 },
    0x20000000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000001, ecx: 0x00000000,  edx: 0x00000000 },
    0x80000000_00000000u64 => CpuIdResult { eax: 0x80000008, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x80000001_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000121,  edx: 0x2c100800 },
    0x80000002_00000000u64 => CpuIdResult { eax: 0x68743231, ebx: 0x6e654720, ecx: 0x746e4920,  edx: 0x52286c65 },
    0x80000003_00000000u64 => CpuIdResult { eax: 0x6f432029, ebx: 0x54286572, ecx: 0x6920294d,  edx: 0x32312d37 },
    0x80000004_00000000u64 => CpuIdResult { eax: 0x4b303037, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x80000005_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x80000006_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x05007040,  edx: 0x00000000 },
    0x80000007_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000100 },
    0x80000008_00000000u64 => CpuIdResult { eax: 0x0000302e, ebx: 0x00000000, ecx: 0x00000000,  edx: 0x00000000 },
    0x80860000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000001, ecx: 0x00000000,  edx: 0x00000000 },
    0xc0000000_00000000u64 => CpuIdResult { eax: 0x00000000, ebx: 0x00000001, ecx: 0x00000000,  edx: 0x00000000 },
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
    assert_eq!(v.as_str(), "GenuineIntel");
}

/// Check feature info gives correct values for CPU
#[test]
fn version_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let f = cpuid.get_feature_info().expect("Need to find feature info");

    assert_eq!(f.base_family_id(), 6);
    assert_eq!(f.base_model_id(), 7);
    assert_eq!(f.stepping_id(), 2);
    assert_eq!(f.extended_family_id(), 0);
    assert_eq!(f.extended_model_id(), 9);
    assert_eq!(f.family_id(), 6);
    assert_eq!(f.model_id(), 151);

    assert_eq!(f.max_logical_processor_ids(), 128);
    assert_eq!(f.initial_local_apic_id(), 0);
    assert_eq!(f.cflush_cache_line_size(), 0x8);
    assert_eq!(f.brand_index(), 0x0);

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
    assert!(f.has_ds());
    assert!(f.has_acpi());
    assert!(f.has_mmx());
    assert!(f.has_fxsave_fxstor());
    assert!(f.has_sse());
    assert!(f.has_sse2());
    assert!(f.has_ss());
    assert!(f.has_htt());
    assert!(f.has_tm());
    assert!(f.has_pbe());

    assert!(f.has_sse3());
    assert!(f.has_pclmulqdq());
    assert!(f.has_ds_area());
    assert!(f.has_monitor_mwait());
    assert!(f.has_cpl());
    assert!(f.has_vmx());
    assert!(f.has_smx());
    assert!(f.has_eist());
    assert!(f.has_tm2());
    assert!(f.has_ssse3());
    assert!(!f.has_cnxtid());
    // has_SDBG
    assert!(f.has_fma());
    assert!(f.has_cmpxchg16b());
    // xTPR
    assert!(f.has_pdcm());
    assert!(f.has_pcid());
    assert!(!f.has_dca());
    assert!(f.has_sse41());
    assert!(f.has_sse42());
    assert!(f.has_x2apic());
    assert!(f.has_movbe());
    assert!(f.has_popcnt());
    assert!(f.has_tsc_deadline());
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
    let ci = cpuid.get_cache_info().expect("Leaf is supported");

    for (idx, cache) in ci.enumerate() {
        match idx {
            0 => assert_eq!(cache.num, 0xf0),
            1 => assert_eq!(cache.num, 0xff),
            2 => assert_eq!(cache.num, 0xfe),
            3 => assert_eq!(cache.num, 0x03),
            4 => assert_eq!(cache.num, 0xf0),
            5 => assert_eq!(cache.num, 0x76),
            6 => assert_eq!(cache.num, 0xc3),
            _ => unreachable!(),
        }
    }
}

#[test]
fn processor_serial() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let psn = cpuid.get_processor_serial().expect("Leaf is supported");
    assert_eq!(psn.serial_lower(), 0x0);
    assert_eq!(psn.serial_middle(), 0x0);
}

#[test]
fn monitor_mwait() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mw = cpuid.get_monitor_mwait_info().expect("Leaf is supported");
    assert_eq!(mw.largest_monitor_line(), 64);
    assert_eq!(mw.smallest_monitor_line(), 64);
    assert!(mw.interrupts_as_break_event());
    assert!(mw.extensions_supported());

    assert_eq!(mw.supported_c0_states(), 0x0);
    assert_eq!(mw.supported_c1_states(), 0x2);
    assert_eq!(mw.supported_c2_states(), 0x0);
    assert_eq!(mw.supported_c3_states(), 0x2);
    assert_eq!(mw.supported_c4_states(), 0x0);
    assert_eq!(mw.supported_c5_states(), 0x1);
    assert_eq!(mw.supported_c6_states(), 0x0);
    assert_eq!(mw.supported_c7_states(), 0x1);
}

#[test]
fn thermal_power() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mw = cpuid.get_thermal_power_info().expect("Leaf is supported");

    assert!(mw.has_dts());
    assert!(mw.has_turbo_boost());
    assert!(mw.has_arat());
    assert!(mw.has_pln());
    assert!(mw.has_ecmd());
    assert!(mw.has_ptm());
    assert!(mw.has_hwp());
    assert!(mw.has_hwp_notification());
    assert!(mw.has_hwp_activity_window());
    assert!(mw.has_hwp_energy_performance_preference());
    assert!(mw.has_hwp_package_level_request());
    assert!(!mw.has_hdc());
    assert!(mw.has_turbo_boost3());
    assert!(mw.has_hwp_capabilities());
    assert!(mw.has_hwp_peci_override());
    assert!(mw.has_flexible_hwp());
    assert!(mw.has_hwp_fast_access_mode());
    assert!(mw.has_hw_coord_feedback());
    assert!(mw.has_ignore_idle_processor_hwp_request());
    // some missing
    assert_eq!(mw.dts_irq_threshold(), 0x2);
    // some missing
    assert!(!mw.has_energy_bias_pref());
}

#[test]
fn extended_features() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_extended_feature_info()
        .expect("Leaf is supported");

    assert!(e.has_fsgsbase());
    assert!(e.has_tsc_adjust_msr());
    assert!(!e.has_sgx());
    assert!(e.has_bmi1());
    assert!(!e.has_hle());
    assert!(e.has_avx2());
    assert!(e.has_fdp());
    assert!(e.has_smep());
    assert!(e.has_bmi2());
    assert!(e.has_rep_movsb_stosb());
    assert!(e.has_invpcid());
    assert!(!e.has_rtm());
    assert!(!e.has_rdtm());
    assert!(e.has_fpu_cs_ds_deprecated());
    assert!(!e.has_mpx());
    assert!(!e.has_rdta());
    assert!(!e.has_avx512f());
    assert!(!e.has_avx512dq());
    assert!(e.has_rdseed());
    assert!(e.has_adx());
    assert!(e.has_smap());
    assert!(!e.has_avx512_ifma());
    assert!(e.has_clflushopt());
    assert!(e.has_clwb());
    assert!(e.has_processor_trace());
    assert!(!e.has_avx512pf());
    assert!(!e.has_avx512er());
    assert!(!e.has_avx512cd());
    assert!(e.has_sha());
    assert!(!e.has_avx512bw());
    assert!(!e.has_avx512vl());
    assert!(!e.has_prefetchwt1());
    // ...
    assert!(e.has_umip());
    assert!(e.has_pku());
    assert!(e.has_ospke());
    assert!(!e.has_avx512vnni());
    assert!(e.has_rdpid());
    assert!(e.has_waitpkg());
    assert!(!e.has_sgx_lc());
    assert_eq!(e.mawau_value(), 0x0);
}

#[test]
fn direct_cache_access() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let dca = cpuid.get_direct_cache_access_info().expect("Leaf exists");
    assert_eq!(dca.get_dca_cap_value(), 0x0);
}

#[test]
fn perfmon_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let pm = cpuid
        .get_performance_monitoring_info()
        .expect("Leaf exists");

    assert_eq!(pm.version_id(), 0x5);

    assert_eq!(pm.number_of_counters(), 0x6);
    assert_eq!(pm.counter_bit_width(), 0x30);
    assert_eq!(pm.ebx_length(), 0x7);

    assert!(!pm.is_core_cyc_ev_unavailable());
    assert!(!pm.is_inst_ret_ev_unavailable());
    assert!(!pm.is_ref_cycle_ev_unavailable());
    assert!(!pm.is_cache_ref_ev_unavailable());
    assert!(!pm.is_ll_cache_miss_ev_unavailable());
    assert!(!pm.is_branch_inst_ret_ev_unavailable());
    assert!(!pm.is_branch_midpred_ev_unavailable());

    assert_eq!(pm.fixed_function_counters(), 0x3);
    assert_eq!(pm.fixed_function_counters_bit_width(), 0x30);
    assert!(pm.has_any_thread_deprecation());
}

#[test]
fn extended_topology_info() {
    use crate::TopologyType;

    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mut e = cpuid
        .get_extended_topology_info()
        .expect("Leaf is supported");

    let t = e.next().expect("Have level 0");
    assert_eq!(t.x2apic_id(), 0);
    assert_eq!(t.level_number(), 0);
    assert_eq!(t.level_type(), TopologyType::SMT);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x1);
    assert_eq!(t.processors(), 2);

    let t = e.next().expect("Have level 1");
    assert_eq!(t.level_number(), 1);
    assert_eq!(t.level_type(), TopologyType::Core);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x7);
    assert_eq!(t.processors(), 20);
    assert_eq!(t.x2apic_id(), 0);
}

#[test]
fn extended_topology_info_v2() {
    use crate::TopologyType;

    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mut e = cpuid
        .get_extended_topology_info_v2()
        .expect("Leaf is supported");

    let t = e.next().expect("Have level 0");
    assert_eq!(t.x2apic_id(), 0);
    assert_eq!(t.level_number(), 0);
    assert_eq!(t.level_type(), TopologyType::SMT);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x1);
    assert_eq!(t.processors(), 2);

    let t = e.next().expect("Have level 1");
    assert_eq!(t.level_number(), 1);
    assert_eq!(t.level_type(), TopologyType::Core);
    assert_eq!(t.shift_right_for_next_apic_id(), 0x7);
    assert_eq!(t.processors(), 20);
    assert_eq!(t.x2apic_id(), 0);
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
    // cpuid binary says this isn't supported, I think it's a bug there and it's
    // supposed to read from ecx1 like we do:
    assert!(e.ia32_xss_supports_pt());
    assert!(e.xcr0_supports_pkru());
    // ...
    assert!(!e.ia32_xss_supports_hdc());

    assert_eq!(e.xsave_area_size_enabled_features(), 2696);
    assert_eq!(e.xsave_area_size_supported_features(), 2696);
    assert!(e.has_xsaveopt());
    assert!(e.has_xsavec());
    assert!(e.has_xgetbv());
    assert!(e.has_xsaves_xrstors());
    // ...
    assert_eq!(e.xsave_size(), 1648);
    // ...

    let mut e = e.iter();
    let ee = e.next().expect("Has level 2");
    assert_eq!(ee.size(), 256);
    assert_eq!(ee.offset(), 576);
    assert!(ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 3");
    assert_eq!(ee.size(), 128);
    assert_eq!(ee.offset(), 0);
    assert!(!ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 4");
    assert_eq!(ee.size(), 8);
    assert_eq!(ee.offset(), 2688);
    assert!(ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 5");
    assert_eq!(ee.size(), 16);
    assert_eq!(ee.offset(), 0);
    assert!(!ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 6");
    assert_eq!(ee.size(), 24);
    assert_eq!(ee.offset(), 0);
    assert!(!ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 7");
    assert_eq!(ee.size(), 808);
    assert_eq!(ee.offset(), 0);
    assert!(!ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());

    let ee = e.next().expect("Has level 8");
    assert_eq!(ee.size(), 8);
    assert_eq!(ee.offset(), 0);
    assert!(!ee.is_in_xcr0());
    assert!(!ee.is_compacted_format());
}

#[test]
fn rdt_monitoring_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_rdt_monitoring_info().expect("Leaf is supported");

    assert_eq!(e.rmid_range(), 0);
    assert!(!e.has_l3_monitoring());
}

#[test]
fn rdt_allocation_info() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_rdt_allocation_info().expect("Leaf is supported");

    assert!(!e.has_l3_cat());
    assert!(!e.has_l2_cat());
    assert!(!e.has_memory_bandwidth_allocation());

    assert!(e.l2_cat().is_none());
    assert!(e.l3_cat().is_none());
}

#[test]
fn sgx_test() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(cpuid.get_sgx_info().is_none());
}

#[test]
fn processor_trace() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let pt = cpuid.get_processor_trace_info().expect("Leaf is available");

    assert!(pt.has_rtit_cr3_match());
    assert!(pt.has_configurable_psb_and_cycle_accurate_mode());
    assert!(pt.has_ip_tracestop_filtering());
    assert!(pt.has_mtc_timing_packet_coefi_suppression());
    assert!(pt.has_ptwrite());
    assert!(!pt.has_power_event_trace());
    assert!(pt.has_topa());
    assert!(pt.has_topa_maximum_entries());
    assert!(pt.has_single_range_output_scheme());
    assert!(!pt.has_trace_transport_subsystem());
    assert!(!pt.has_lip_with_cs_base());

    assert_eq!(pt.configurable_address_ranges(), 2);
    assert_eq!(pt.supported_mtc_period_encodings(), 585);
    assert_eq!(pt.supported_cycle_threshold_value_encodings(), 63);
    assert_eq!(pt.supported_psb_frequency_encodings(), 63);
}

#[test]
fn tsc() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_tsc_info().expect("Leaf is available");
    assert_eq!(e.denominator(), 2);
    assert_eq!(e.numerator(), 188);
    assert_eq!(e.nominal_frequency(), 38400000);
    assert_eq!(e.tsc_frequency(), Some(3609600000));
}

#[test]
fn processor_frequency() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_processor_frequency_info()
        .expect("Leaf is supported");

    assert_eq!(e.processor_base_frequency(), 3600);
    assert_eq!(e.processor_max_frequency(), 5000);
    assert_eq!(e.bus_frequency(), 100);
}

#[test]
fn extended_processor_and_feature_identifiers() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_extended_processor_and_feature_identifiers()
        .expect("Leaf is supported");

    assert_eq!(e.pkg_type(), 0x0); // reserved on Intel
    assert_eq!(e.brand_id(), 0x0); // reserved on Intel

    assert!(e.has_lahf_sahf());
    assert!(!e.has_cmp_legacy());
    assert!(!e.has_svm());
    assert!(!e.has_ext_apic_space());
    assert!(!e.has_alt_mov_cr8());
    assert!(e.has_lzcnt());
    assert!(!e.has_sse4a());
    assert!(!e.has_misaligned_sse_mode());
    assert!(e.has_prefetchw());
    assert!(!e.has_osvw());
    assert!(!e.has_ibs());
    assert!(!e.has_xop());
    assert!(!e.has_skinit());
    assert!(!e.has_wdt());
    assert!(!e.has_lwp());
    assert!(!e.has_fma4());
    assert!(!e.has_tbm());
    assert!(!e.has_topology_extensions());
    assert!(!e.has_perf_cntr_extensions());
    assert!(!e.has_nb_perf_cntr_extensions());
    assert!(!e.has_data_access_bkpt_extension());
    assert!(!e.has_perf_tsc());
    assert!(!e.has_perf_cntr_llc_extensions());
    assert!(!e.has_monitorx_mwaitx());
    assert!(!e.has_addr_mask_extension());
    assert!(e.has_syscall_sysret());
    assert!(e.has_execute_disable());
    assert!(!e.has_mmx_extensions());
    assert!(!e.has_fast_fxsave_fxstor());
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

    assert_eq!(e.as_str(), "12th Gen Intel(R) Core(TM) i7-12700K");
}

#[test]
fn l1_tlb_cache() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    assert!(cpuid.get_l1_cache_and_tlb_info().is_none());
}

#[test]
fn l2_l3_tlb_cache() {
    use crate::Associativity;

    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_l2_l3_cache_and_tlb_info()
        .expect("Leaf is supported");

    // Unsupported on Intel
    assert_eq!(e.itlb_2m_4m_associativity(), Associativity::Disabled);
    assert_eq!(e.itlb_2m_4m_size(), 0);
    assert_eq!(e.dtlb_2m_4m_associativity(), Associativity::Disabled);
    assert_eq!(e.dtlb_2m_4m_size(), 0);
    assert_eq!(e.itlb_4k_size(), 0);
    assert_eq!(e.itlb_4k_associativity(), Associativity::Disabled);
    assert_eq!(e.dtlb_4k_size(), 0);
    assert_eq!(e.dtlb_4k_associativity(), Associativity::Disabled);

    // Supported on Intel
    assert_eq!(e.l2cache_line_size(), 64);
    assert_eq!(e.l2cache_lines_per_tag(), 0);
    assert_eq!(e.l2cache_associativity(), Associativity::Unknown);
    assert_eq!(e.l2cache_size(), 1280);

    // Unsupported on Intel
    assert_eq!(e.l3cache_line_size(), 0);
    assert_eq!(e.l3cache_lines_per_tag(), 0);
    assert_eq!(e.l3cache_associativity(), Associativity::Disabled);
    assert_eq!(e.l3cache_size(), 0);
}

#[test]
fn apm() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_advanced_power_mgmt_info()
        .expect("Leaf is supported");

    assert!(!e.has_mca_overflow_recovery());
    assert!(!e.has_succor());
    assert!(!e.has_hwa());
    // ...
    assert_eq!(e.cpu_pwr_sample_time_ratio(), 0x0);

    assert!(!e.has_ts());
    assert!(!e.has_freq_id_ctrl());
    assert!(!e.has_volt_id_ctrl());
    assert!(!e.has_thermtrip());
    assert!(!e.has_tm());
    assert!(!e.has_100mhz_steps());
    assert!(!e.has_hw_pstate());
    assert!(e.has_invariant_tsc()); // The only Intel supported feature here
    assert!(!e.has_cpb());
    assert!(!e.has_ro_effective_freq_iface());
    assert!(!e.has_feedback_iface());
    assert!(!e.has_power_reporting_iface());
}

#[test]
fn processor_capcity_features() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid
        .get_processor_capacity_feature_info()
        .expect("Leaf is supported");

    assert_eq!(e.physical_address_bits(), 46);
    assert_eq!(e.linear_address_bits(), 48);
    assert_eq!(e.guest_physical_address_bits(), 0);

    assert!(!e.has_cl_zero());
    assert!(!e.has_inst_ret_cntr_msr());
    assert!(!e.has_restore_fp_error_ptrs());
    assert!(!e.has_invlpgb());
    assert!(!e.has_rdpru());
    assert!(!e.has_mcommit());
    assert!(!e.has_wbnoinvd());
    assert!(!e.has_int_wbinvd());
    assert!(!e.has_unsupported_efer_lmsle());
    assert!(!e.has_invlpgb_nested());

    assert_eq!(e.invlpgb_max_pages(), 0x0);

    assert_eq!(e.maximum_logical_processors(), 1); // Not sure why this is set, it's reserved :(
    assert_eq!(e.num_phys_threads(), 1); // Not sure why this is set, it's reserved :(
    assert_eq!(e.apic_id_size(), 0);
    assert_eq!(e.perf_tsc_size(), 40); // Not sure why this is set, it's reserved :(
    assert_eq!(e.max_rdpru_id(), 0);
}

#[test]
fn get_deterministic_address_translation_info() {
    use crate::DatType;

    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let mut e = cpuid
        .get_deterministic_address_translation_info()
        .expect("Leaf is supported");

    // This is a null entry, so all of this should be 0/false/invalid/null
    let t = e.next().expect("Have level 1");
    assert!(t.has_4k_entries());
    assert!(!t.has_2mb_entries());
    assert!(!t.has_4mb_entries());
    assert!(!t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 8);
    assert_eq!(t.sets(), 32);
    assert_eq!(t.cache_type(), DatType::InstructionTLB);
    assert_eq!(t.cache_level(), 1);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 2");
    assert!(!t.has_4k_entries());
    assert!(t.has_2mb_entries());
    assert!(t.has_4mb_entries());
    assert!(!t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 8);
    assert_eq!(t.sets(), 4);
    assert_eq!(t.cache_type(), DatType::InstructionTLB);
    assert_eq!(t.cache_level(), 1);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 3");
    assert!(t.has_4k_entries());
    assert!(t.has_2mb_entries());
    assert!(t.has_4mb_entries());
    assert!(t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 16);
    assert_eq!(t.sets(), 1);
    assert_eq!(t.cache_type(), DatType::StoreOnly);
    assert_eq!(t.cache_level(), 1);
    assert!(t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 4");
    assert!(t.has_4k_entries());
    assert!(!t.has_2mb_entries());
    assert!(!t.has_4mb_entries());
    assert!(!t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 4);
    assert_eq!(t.sets(), 16);
    assert_eq!(t.cache_type(), DatType::LoadOnly);
    assert_eq!(t.cache_level(), 1);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 5");
    assert!(!t.has_4k_entries());
    assert!(t.has_2mb_entries());
    assert!(t.has_4mb_entries());
    assert!(!t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 4);
    assert_eq!(t.sets(), 8);
    assert_eq!(t.cache_type(), DatType::LoadOnly);
    assert_eq!(t.cache_level(), 1);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 6");
    assert!(!t.has_4k_entries());
    assert!(!t.has_2mb_entries());
    assert!(!t.has_4mb_entries());
    assert!(t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 8);
    assert_eq!(t.sets(), 1);
    assert_eq!(t.cache_type(), DatType::LoadOnly);
    assert_eq!(t.cache_level(), 1);
    assert!(t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 7");
    assert!(t.has_4k_entries());
    assert!(t.has_2mb_entries());
    assert!(t.has_4mb_entries());
    assert!(!t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 8);
    assert_eq!(t.sets(), 128);
    assert_eq!(t.cache_type(), DatType::UnifiedTLB);
    assert_eq!(t.cache_level(), 2);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);

    let t = e.next().expect("Have level 8");
    assert!(t.has_4k_entries());
    assert!(!t.has_2mb_entries());
    assert!(!t.has_4mb_entries());
    assert!(t.has_1gb_entries());
    assert_eq!(t.partitioning(), 0);
    assert_eq!(t.ways(), 8);
    assert_eq!(t.sets(), 128);
    assert_eq!(t.cache_type(), DatType::UnifiedTLB);
    assert_eq!(t.cache_level(), 2);
    assert!(!t.is_fully_associative());
    assert_eq!(t.max_addressable_ids(), 2);
}

#[test]
fn get_soc_vendor() {
    let cpuid = CpuId::with_cpuid_fn(cpuid_reader);
    let e = cpuid.get_soc_vendor_info().expect("Leaf is supported");

    assert_eq!(e.get_project_id(), 0);
    assert_eq!(e.get_soc_vendor_id(), 0);
    assert_eq!(e.get_stepping_id(), 0);

    if let Some(attr_iter) = e.get_vendor_attributes() {
        for attr in attr_iter {
            println!("{:?}", attr);
        }
    }

    println!("{:?}", e.get_vendor_brand());
}
