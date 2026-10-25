use std::fmt::Display;

use crate::{
    Associativity, CacheType, CpuIdResult, DatType, ExtendedRegisterStateLocation, SgxSectionInfo,
    SoCVendorBrand, TopologyType,
};

use termimad::{minimad::TextTemplate, minimad::TextTemplateExpander, MadSkin};

pub fn raw<R: crate::CpuIdReader>(cpuid: R) {
    let _leafs_with_subleafs = &[0x04, 0x0d, 0x0f, 0x10, 0x12];

    let max_leafs = cpuid.cpuid1(0x0).eax;
    for idx in 0..max_leafs {
        let res = cpuid.cpuid1(idx);
        println!("({:#x}, {:#x}) => {:?}", idx, 0x0, res);
    }

    let max_hypervisor_leafs = cpuid.cpuid1(0x4000_0000).eax;
    for idx in 0x4000_0000..max_hypervisor_leafs {
        println!("({:#x}, {:#x}) => {:?}", idx, 0x0, cpuid.cpuid1(idx));
    }

    let max_extended_leafs = cpuid.cpuid1(0x8000_0000).eax;
    for idx in 0x8000_0000..max_extended_leafs {
        println!("({:#x}, {:#x}) => {:?}", idx, 0x0, cpuid.cpuid1(idx));
    }
}

fn table2(skin: &MadSkin, attrs: &[(&str, String)]) {
    let table_template = TextTemplate::from(
        r#"
|-:|-:|
${feature-rows
|**${attr-name}**|${attr-avail}|
}
|-|-|
    "#,
    );

    fn make_table_display<'a, 'b>(
        text_template: &'a TextTemplate<'b>,
        attrs: &'b [(&'b str, String)],
    ) -> TextTemplateExpander<'a, 'b> {
        let mut expander = text_template.expander();

        for (attr, desc) in attrs {
            expander
                .sub("feature-rows")
                .set("attr-name", attr)
                .set("attr-avail", desc);
        }

        expander
    }

    let table = make_table_display(&table_template, &attrs);
    skin.print_expander(table);
}

fn table3(skin: &MadSkin, attrs: &[(&'static str, &'static str, String)]) {
    let table_template3 = TextTemplate::from(
        r#"
|:-|-:|-:|
${feature-rows
|**${category-name}**|**${attr-name}**|${attr-avail}|
}
|-|-|
    "#,
    );

    fn make_table_display3<'a, 'b>(
        text_template: &'a TextTemplate<'b>,
        attrs: &'b [(&'b str, &'b str, String)],
    ) -> TextTemplateExpander<'a, 'b> {
        let mut expander = text_template.expander();

        for (cat, attr, desc) in attrs {
            expander
                .sub("feature-rows")
                .set("category-name", cat)
                .set("attr-name", attr)
                .set("attr-avail", desc);
        }

        expander
    }

    let table = make_table_display3(&table_template3, &attrs);
    skin.print_expander(table);
}

fn print_title_line(skin: &MadSkin, title: &str, attr: Option<&str>) {
    if let Some(opt) = attr {
        skin.print_text(format!("## {} = \"{}\"\n", title, opt).as_str());
    } else {
        skin.print_text(format!("## {}\n", title).as_str());
    }
}

fn print_title_attr(skin: &MadSkin, title: &str, attr: &str) {
    print_title_line(skin, title, Some(attr));
}

fn print_title(skin: &MadSkin, title: &str) {
    print_title_line(skin, title, None)
}

fn print_subtitle(skin: &MadSkin, title: &str) {
    skin.print_text(format!("### {}\n", title).as_str());
}

fn print_attr<T: Display, A: Display>(skin: &MadSkin, name: T, attr: A) {
    skin.print_text(format!("{} = {}", name, attr).as_str());
}

fn print_cpuid_result<T: Display>(skin: &MadSkin, name: T, attr: CpuIdResult) {
    skin.print_text(
        format!(
            "{}: eax = {:#x} ebx = {:#x} ecx = {:#x} edx = {:#x}",
            name, attr.eax, attr.ebx, attr.ecx, attr.edx,
        )
        .as_str(),
    );
}

fn bool_repr(x: bool) -> String {
    if x {
        "✅".to_string()
    } else {
        "❌".to_string()
    }
}

trait RowGen {
    fn fmt(attr: &Self) -> String;

    fn tuple<'a>(t: &'a str, attr: Self) -> (&'a str, String)
    where
        Self: Sized,
    {
        (t, RowGen::fmt(&attr))
    }

    fn triple<'a>(c: &'a str, t: &'a str, attr: Self) -> (&'a str, &'a str, String)
    where
        Self: Sized,
    {
        (c, t, RowGen::fmt(&attr))
    }
}

impl RowGen for bool {
    fn fmt(attr: &Self) -> String {
        bool_repr(*attr)
    }
}

impl RowGen for u64 {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for usize {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for u32 {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for u16 {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for u8 {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for String {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for Associativity {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for CacheType {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for TopologyType {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for ExtendedRegisterStateLocation {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for DatType {
    fn fmt(attr: &Self) -> String {
        format!("{}", attr)
    }
}

impl RowGen for Option<SoCVendorBrand> {
    fn fmt(attr: &Self) -> String {
        format!(
            "{}",
            attr.as_ref()
                .map(|v| v.as_str().to_string())
                .unwrap_or(String::from(""))
        )
    }
}

pub fn markdown<R: crate::CpuIdReader>(cpuid: crate::CpuId<R>) {
    let skin = MadSkin::default();
    skin.print_text("# CpuId\n");

    if let Some(info) = cpuid.get_vendor_info() {
        print_title_attr(&skin, "vendor_id (0x00)", info.as_str());
    }

    if let Some(info) = cpuid.get_feature_info() {
        print_title(&skin, "version information (1/eax):");
        table2(
            &skin,
            &[
                RowGen::tuple("base family", info.base_family_id()),
                RowGen::tuple("base model", info.base_model_id()),
                RowGen::tuple("stepping", info.stepping_id()),
                RowGen::tuple("extended family", info.extended_family_id()),
                RowGen::tuple("extended model", info.extended_model_id()),
                RowGen::tuple("family", info.family_id()),
                RowGen::tuple("model", info.model_id()),
            ],
        );

        print_title(&skin, "miscellaneous (1/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple("processor APIC physical id", info.initial_local_apic_id()),
                RowGen::tuple("max. cpus", info.max_logical_processor_ids()),
                RowGen::tuple("CLFLUSH line size", info.cflush_cache_line_size()),
                RowGen::tuple("brand index", info.brand_index()),
            ],
        );

        print_title(&skin, "feature information (1/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("fpu", info.has_fpu()),
                RowGen::tuple("vme", info.has_vme()),
                RowGen::tuple("de", info.has_de()),
                RowGen::tuple("pse", info.has_pse()),
                RowGen::tuple("tsc", info.has_tsc()),
                RowGen::tuple("msr", info.has_msr()),
                RowGen::tuple("pae", info.has_pae()),
                RowGen::tuple("mce", info.has_mce()),
                RowGen::tuple("cmpxchg8b", info.has_cmpxchg8b()),
                RowGen::tuple("apic", info.has_apic()),
                RowGen::tuple("sysenter_sysexit", info.has_sysenter_sysexit()),
                RowGen::tuple("mtrr", info.has_mtrr()),
                RowGen::tuple("pge", info.has_pge()),
                RowGen::tuple("mca", info.has_mca()),
                RowGen::tuple("cmov", info.has_cmov()),
                RowGen::tuple("pat", info.has_pat()),
                RowGen::tuple("pse36", info.has_pse36()),
                RowGen::tuple("psn", info.has_psn()),
                RowGen::tuple("clflush", info.has_clflush()),
                RowGen::tuple("ds", info.has_ds()),
                RowGen::tuple("acpi", info.has_acpi()),
                RowGen::tuple("mmx", info.has_mmx()),
                RowGen::tuple("fxsave_fxstor", info.has_fxsave_fxstor()),
                RowGen::tuple("sse", info.has_sse()),
                RowGen::tuple("sse2", info.has_sse2()),
                RowGen::tuple("ss", info.has_ss()),
                RowGen::tuple("htt", info.has_htt()),
                RowGen::tuple("tm", info.has_tm()),
                RowGen::tuple("pbe", info.has_pbe()),
            ],
        );

        print_title(&skin, "feature information (1/ecx):");
        table2(
            &skin,
            &[
                RowGen::tuple("sse3", info.has_sse3()),
                RowGen::tuple("pclmulqdq", info.has_pclmulqdq()),
                RowGen::tuple("ds_area", info.has_ds_area()),
                RowGen::tuple("monitor_mwait", info.has_monitor_mwait()),
                RowGen::tuple("cpl", info.has_cpl()),
                RowGen::tuple("vmx", info.has_vmx()),
                RowGen::tuple("smx", info.has_smx()),
                RowGen::tuple("eist", info.has_eist()),
                RowGen::tuple("tm2", info.has_tm2()),
                RowGen::tuple("ssse3", info.has_ssse3()),
                RowGen::tuple("cnxtid", info.has_cnxtid()),
                RowGen::tuple("fma", info.has_fma()),
                RowGen::tuple("cmpxchg16b", info.has_cmpxchg16b()),
                RowGen::tuple("pdcm", info.has_pdcm()),
                RowGen::tuple("pcid", info.has_pcid()),
                RowGen::tuple("dca", info.has_dca()),
                RowGen::tuple("sse41", info.has_sse41()),
                RowGen::tuple("sse42", info.has_sse42()),
                RowGen::tuple("x2apic", info.has_x2apic()),
                RowGen::tuple("movbe", info.has_movbe()),
                RowGen::tuple("popcnt", info.has_popcnt()),
                RowGen::tuple("tsc_deadline", info.has_tsc_deadline()),
                RowGen::tuple("aesni", info.has_aesni()),
                RowGen::tuple("xsave", info.has_xsave()),
                RowGen::tuple("oxsave", info.has_oxsave()),
                RowGen::tuple("avx", info.has_avx()),
                RowGen::tuple("f16c", info.has_f16c()),
                RowGen::tuple("rdrand", info.has_rdrand()),
                RowGen::tuple("hypervisor", info.has_hypervisor()),
            ],
        );
    }

    if let Some(info) = cpuid.get_cache_info() {
        print_title(&skin, "Cache and TLB information (0x02):");
        let nums: Vec<String> = info
            .clone()
            .map(|cache| format!("{:#x}", cache.num))
            .collect();
        let attrs: Vec<(&str, String)> = info
            .zip(nums.iter())
            .map(|(cache, n)| RowGen::tuple(n, cache.desc().to_string()))
            .collect();
        table2(&skin, &attrs);
    }

    if let Some(info) = cpuid.get_processor_serial() {
        print_title_attr(
            &skin,
            "processor serial number (0x03)",
            format!(
                "{:0>8x}-{:0>8x}-{:0>8x}",
                info.serial_upper(),
                info.serial_middle(),
                info.serial_lower()
            )
            .as_str(),
        );
    }

    if let Some(iter) = cpuid.get_cache_parameters() {
        print_title(&skin, "deterministic cache parameters (0x04):");
        for cache in iter {
            print_subtitle(&skin, format!("L{} Cache:", cache.level()).as_str());

            let size = (cache.associativity()
                * cache.physical_line_partitions()
                * cache.coherency_line_size()
                * cache.sets()) as u64;

            table2(
                &skin,
                &[
                    RowGen::tuple("cache type", cache.cache_type()),
                    RowGen::tuple("cache level", cache.level()),
                    RowGen::tuple(
                        "self-initializing cache level",
                        cache.is_self_initializing(),
                    ),
                    RowGen::tuple("fully associative cache", cache.is_fully_associative()),
                    RowGen::tuple("threads sharing this cache", cache.max_cores_for_cache()),
                    RowGen::tuple("processor cores on this die", cache.max_cores_for_package()),
                    RowGen::tuple("system coherency line size", cache.coherency_line_size()),
                    RowGen::tuple("physical line partitions", cache.physical_line_partitions()),
                    RowGen::tuple("ways of associativity", cache.associativity()),
                    RowGen::tuple(
                        "WBINVD/INVD acts on lower caches",
                        cache.is_write_back_invalidate(),
                    ),
                    RowGen::tuple("inclusive to lower caches", cache.is_inclusive()),
                    RowGen::tuple("complex cache indexing", cache.has_complex_indexing()),
                    RowGen::tuple("number of sets", cache.sets()),
                    RowGen::tuple("(size synth.)", size),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_monitor_mwait_info() {
        print_title(&skin, "MONITOR/MWAIT (0x05):");
        table2(
            &skin,
            &[
                RowGen::tuple("smallest monitor-line size", info.smallest_monitor_line()),
                RowGen::tuple("largest monitor-line size", info.largest_monitor_line()),
                RowGen::tuple("MONITOR/MWAIT exts", info.extensions_supported()),
                RowGen::tuple(
                    "Interrupts as break-event for MWAIT",
                    info.interrupts_as_break_event(),
                ),
            ],
        );

        skin.print_text("number of CX sub C-states using MWAIT:\n");
        let cstate_table = TextTemplate::from(
            r#"
        | :-: |  :-: | :-: | :-: | :-: | :-: | :-: | :-: |
        |**C0**|**C1**|**C2**|**C3**|**C4**|**C5**|**C6**|**C7**|
        | :-: |  :-: | :-: | :-: | :-: | :-: | :-: | :-: |
        |${c0}|${c1}|${c2}|${c3}|${c4}|${c5}|${c6}|${c7}|
        | :-: |  :-: | :-: | :-: | :-: | :-: | :-: | :-: |
        "#,
        );
        let c0 = format!("{}", info.supported_c0_states());
        let c1 = format!("{}", info.supported_c1_states());
        let c2 = format!("{}", info.supported_c2_states());
        let c3 = format!("{}", info.supported_c3_states());
        let c4 = format!("{}", info.supported_c4_states());
        let c5 = format!("{}", info.supported_c5_states());
        let c6 = format!("{}", info.supported_c6_states());
        let c7 = format!("{}", info.supported_c7_states());

        let mut ctbl = cstate_table.expander();
        ctbl.set("c0", c0.as_str());
        ctbl.set("c1", c1.as_str());
        ctbl.set("c2", c2.as_str());
        ctbl.set("c3", c3.as_str());
        ctbl.set("c4", c4.as_str());
        ctbl.set("c5", c5.as_str());
        ctbl.set("c6", c6.as_str());
        ctbl.set("c7", c7.as_str());
        skin.print_expander(ctbl);
    }

    if let Some(info) = cpuid.get_thermal_power_info() {
        print_title(&skin, "Thermal and Power Management Features (0x06):");
        table2(
            &skin,
            &[
                RowGen::tuple("digital thermometer", info.has_dts()),
                RowGen::tuple("Intel Turbo Boost Technology", info.has_turbo_boost()),
                RowGen::tuple("ARAT always running APIC timer", info.has_arat()),
                RowGen::tuple("PLN power limit notification", info.has_pln()),
                RowGen::tuple("ECMD extended clock modulation duty", info.has_ecmd()),
                RowGen::tuple("PTM package thermal management", info.has_ptm()),
                RowGen::tuple("HWP base registers", info.has_hwp()),
                RowGen::tuple("HWP notification", info.has_hwp_notification()),
                RowGen::tuple("HWP activity window", info.has_hwp_activity_window()),
                RowGen::tuple(
                    "HWP energy performance preference",
                    info.has_hwp_energy_performance_preference(),
                ),
                RowGen::tuple(
                    "HWP package level request",
                    info.has_hwp_package_level_request(),
                ),
                RowGen::tuple("HDC base registers", info.has_hdc()),
                RowGen::tuple(
                    "Intel Turbo Boost Max Technology 3.0",
                    info.has_turbo_boost3(),
                ),
                RowGen::tuple("HWP capabilities", info.has_hwp_capabilities()),
                RowGen::tuple("HWP PECI override", info.has_hwp_peci_override()),
                RowGen::tuple("flexible HWP", info.has_flexible_hwp()),
                RowGen::tuple(
                    "IA32_HWP_REQUEST MSR fast access mode",
                    info.has_hwp_fast_access_mode(),
                ),
                RowGen::tuple(
                    "ignoring idle logical processor HWP req",
                    info.has_ignore_idle_processor_hwp_request(),
                ),
                RowGen::tuple("digital thermometer threshold", info.dts_irq_threshold()),
                RowGen::tuple(
                    "hardware coordination feedback",
                    info.has_hw_coord_feedback(),
                ),
                RowGen::tuple(
                    "performance-energy bias capability",
                    info.has_energy_bias_pref(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_extended_feature_info() {
        print_title(&skin, "Extended feature flags (0x07):");

        table2(
            &skin,
            &[
                RowGen::tuple("FSGSBASE", info.has_fsgsbase()),
                RowGen::tuple("IA32_TSC_ADJUST MSR", info.has_tsc_adjust_msr()),
                RowGen::tuple("SGX: Software Guard Extensions", info.has_sgx()),
                RowGen::tuple("BMI1", info.has_bmi1()),
                RowGen::tuple("HLE hardware lock elision", info.has_hle()),
                RowGen::tuple("AVX2: advanced vector extensions 2", info.has_avx2()),
                RowGen::tuple("FDP_EXCPTN_ONLY", info.has_fdp()),
                RowGen::tuple("SMEP supervisor mode exec protection", info.has_smep()),
                RowGen::tuple("BMI2 instructions", info.has_bmi2()),
                RowGen::tuple("enhanced REP MOVSB/STOSB", info.has_rep_movsb_stosb()),
                RowGen::tuple("INVPCID instruction", info.has_invpcid()),
                RowGen::tuple("RTM: restricted transactional memory", info.has_rtm()),
                RowGen::tuple("RDT-CMT/PQoS cache monitoring", info.has_rdtm()),
                RowGen::tuple("deprecated FPU CS/DS", info.has_fpu_cs_ds_deprecated()),
                RowGen::tuple("MPX: intel memory protection extensions", info.has_mpx()),
                RowGen::tuple("RDT-CAT/PQE cache allocation", info.has_rdta()),
                RowGen::tuple(
                    "AVX512F: AVX-512 foundation instructions",
                    info.has_avx512f(),
                ),
                RowGen::tuple("AVX512-4NNIW: 4NNIW instructions", info.has_avx512_4vnniw()),
                RowGen::tuple(
                    "AVX512-4FMAPS: 4FMAPS instructions",
                    info.has_avx512_4fmaps(),
                ),
                RowGen::tuple(
                    "AVX512-VP2INTERSECT: VP2INTERSECT instructions",
                    info.has_avx512_vp2intersect(),
                ),
                RowGen::tuple("AMX_BF16: AMX_BF16 instructions", info.has_amx_bf16()),
                RowGen::tuple(
                    "AVX512_FP16: AVX512_FP16 instructions",
                    info.has_avx512_fp16(),
                ),
                RowGen::tuple("AMX_TILE: Tile Architecture support", info.has_amx_tile()),
                RowGen::tuple(
                    "AMX_INT8: Tile Computational Operation on 8-bit integers",
                    info.has_amx_tile(),
                ),
                RowGen::tuple(
                    "AVX512DQ: double & quadword instructions",
                    info.has_avx512dq(),
                ),
                RowGen::tuple("RDSEED instruction", info.has_rdseed()),
                RowGen::tuple("ADX instructions", info.has_adx()),
                RowGen::tuple("SMAP: supervisor mode access prevention", info.has_smap()),
                RowGen::tuple("AVX512IFMA: fused multiply add", info.has_avx512_ifma()),
                RowGen::tuple("CLFLUSHOPT instruction", info.has_clflushopt()),
                RowGen::tuple("CLWB instruction", info.has_clwb()),
                RowGen::tuple("Intel processor trace", info.has_processor_trace()),
                RowGen::tuple("AVX512PF: prefetch instructions", info.has_avx512pf()),
                RowGen::tuple(
                    "AVX512ER: exponent & reciprocal instrs",
                    info.has_avx512er(),
                ),
                RowGen::tuple("AVX512CD: conflict detection instrs", info.has_avx512cd()),
                RowGen::tuple("SHA instructions", info.has_sha()),
                RowGen::tuple("AVX512BW: byte & word instructions", info.has_avx512bw()),
                RowGen::tuple("AVX512VL: vector length", info.has_avx512vl()),
                RowGen::tuple("PREFETCHWT1", info.has_prefetchwt1()),
                RowGen::tuple("UMIP: user-mode instruction prevention", info.has_umip()),
                RowGen::tuple("PKU protection keys for user-mode", info.has_pku()),
                RowGen::tuple("OSPKE CR4.PKE and RDPKRU/WRPKRU", info.has_ospke()),
                RowGen::tuple(
                    "AVX512VNNI: vector neural network instructions",
                    info.has_avx512vnni(),
                ),
                RowGen::tuple(
                    "BNDLDX/BNDSTX MAWAU value in 64-bit mode",
                    info.mawau_value(),
                ),
                RowGen::tuple("RDPID: read processor ID", info.has_rdpid()),
                RowGen::tuple("SGX_LC: SGX launch config", info.has_sgx_lc()),
                RowGen::tuple(
                    "AVX_VNNI: AVX vector neural network instructions",
                    info.has_avx_vnni(),
                ),
                RowGen::tuple(
                    "AVX512_BF16: AVX512 BF16 instructions",
                    info.has_avx512_bf16(),
                ),
                RowGen::tuple("FZRMK: fast zero-length REP MOVSB256", info.has_fzrm()),
                RowGen::tuple("FSRM: fast short REP STOSB", info.has_fsrs()),
                RowGen::tuple("FSRCRS: fast short REP CMPSB, REP SCASB", info.has_fsrcrs()),
                RowGen::tuple("HRESET: HRESET instruction", info.has_hreset()),
                RowGen::tuple("CET_SSS: CET_SSS support", info.has_cet_sss()),
            ],
        );
    }

    if let Some(info) = cpuid.get_direct_cache_access_info() {
        print_title(&skin, "Direct Cache Access Parameters (0x09):");
        print_attr(&skin, "PLATFORM_DCA_CAP MSR bits", info.get_dca_cap_value());
    }

    if let Some(info) = cpuid.get_performance_monitoring_info() {
        print_title(&skin, "Architecture Performance Monitoring Features (0x0a)");

        print_subtitle(&skin, "Monitoring Hardware Info (0x0a/{eax, edx}):");
        table2(
            &skin,
            &[
                RowGen::tuple("version ID", info.version_id()),
                RowGen::tuple(
                    "number of counters per HW thread",
                    info.number_of_counters(),
                ),
                RowGen::tuple("bit width of counter", info.counter_bit_width()),
                RowGen::tuple("length of EBX bit vector", info.ebx_length()),
                RowGen::tuple("number of fixed counters", info.fixed_function_counters()),
                RowGen::tuple(
                    "bit width of fixed counters",
                    info.fixed_function_counters_bit_width(),
                ),
                RowGen::tuple("anythread deprecation", info.has_any_thread_deprecation()),
            ],
        );

        print_subtitle(&skin, "Monitoring Hardware Features (0x0a/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "core cycle event not available",
                    info.is_core_cyc_ev_unavailable(),
                ),
                RowGen::tuple(
                    "instruction retired event not available",
                    info.is_inst_ret_ev_unavailable(),
                ),
                RowGen::tuple(
                    "reference cycles event not available",
                    info.is_ref_cycle_ev_unavailable(),
                ),
                RowGen::tuple(
                    "last-level cache ref event not available",
                    info.is_cache_ref_ev_unavailable(),
                ),
                RowGen::tuple(
                    "last-level cache miss event not avail",
                    info.is_ll_cache_miss_ev_unavailable(),
                ),
                RowGen::tuple(
                    "branch inst retired event not available",
                    info.is_branch_inst_ret_ev_unavailable(),
                ),
                RowGen::tuple(
                    "branch mispred retired event not available",
                    info.is_branch_midpred_ev_unavailable(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_extended_topology_info() {
        print_title(&skin, "x2APIC features / processor topology (0x0b):");

        for level in info {
            print_subtitle(&skin, format!("level {}:", level.level_number()).as_str());
            table2(
                &skin,
                &[
                    RowGen::tuple("level type", level.level_type()),
                    RowGen::tuple("bit width of level", level.shift_right_for_next_apic_id()),
                    RowGen::tuple("number of logical processors at level", level.processors()),
                    RowGen::tuple("x2apic id of current processor", level.x2apic_id()),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_extended_state_info() {
        print_title(&skin, "Extended Register State (0x0d/0):");

        print_subtitle(&skin, "XCR0/IA32_XSS supported states:");
        table3(
            &skin,
            &[
                RowGen::triple("XCR0", "x87", info.xcr0_supports_legacy_x87()),
                RowGen::triple("XCR0", "SSE state", info.xcr0_supports_sse_128()),
                RowGen::triple("XCR0", "AVX state", info.xcr0_supports_avx_256()),
                RowGen::triple("XCR0", "MPX BNDREGS", info.xcr0_supports_mpx_bndregs()),
                RowGen::triple("XCR0", "MPX BNDCSR", info.xcr0_supports_mpx_bndcsr()),
                RowGen::triple("XCR0", "AVX-512 opmask", info.xcr0_supports_avx512_opmask()),
                RowGen::triple(
                    "XCR0",
                    "AVX-512 ZMM_Hi256",
                    info.xcr0_supports_avx512_zmm_hi256(),
                ),
                RowGen::triple(
                    "XCR0",
                    "AVX-512 Hi16_ZMM",
                    info.xcr0_supports_avx512_zmm_hi16(),
                ),
                RowGen::triple("IA32_XSS", "PT", info.ia32_xss_supports_pt()),
                RowGen::triple("XCR0", "PKRU", info.xcr0_supports_pkru()),
                //("XCR0", "CET_U state", xxx),
                //("XCR0", "CET_S state", xxx),
                RowGen::triple("IA32_XSS", "HDC", info.ia32_xss_supports_hdc()),
            ],
        );

        table2(
            &skin,
            &[
                RowGen::tuple(
                    "bytes required by fields in XCR0",
                    info.xsave_area_size_enabled_features(),
                ),
                RowGen::tuple(
                    "bytes required by XSAVE/XRSTOR area",
                    info.xsave_area_size_supported_features(),
                ),
            ],
        );

        print_subtitle(&skin, "XSAVE features (0x0d/1):");
        table2(
            &skin,
            &[
                RowGen::tuple("XSAVEOPT instruction", info.has_xsaveopt()),
                RowGen::tuple("XSAVEC instruction", info.has_xsavec()),
                RowGen::tuple("XGETBV instruction", info.has_xgetbv()),
                RowGen::tuple("XSAVES/XRSTORS instructions", info.has_xsaves_xrstors()),
                RowGen::tuple("SAVE area size [Bytes]", info.xsave_size()),
            ],
        );

        for state in info.iter() {
            print_subtitle(
                &skin,
                format!("{} features (0x0d/{}):", state.register(), state.subleaf).as_str(),
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("save state size [Bytes]", state.size()),
                    RowGen::tuple("save state byte offset", state.offset()),
                    RowGen::tuple("supported in IA32_XSS or XCR0", state.location()),
                    RowGen::tuple(
                        "64-byte alignment in compacted XSAVE",
                        state.is_compacted_format(),
                    ),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_rdt_monitoring_info() {
        print_title(
            &skin,
            "Quality of Service Monitoring Resource Type (0x0f/0):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple("Maximum range of RMID", info.rmid_range()),
                RowGen::tuple("L3 cache QoS monitoring", info.has_l3_monitoring()),
            ],
        );

        if let Some(rmid) = info.l3_monitoring() {
            print_subtitle(&skin, "L3 Cache Quality of Service Monitoring (0x0f/1):");

            table2(
                &skin,
                &[
                    RowGen::tuple(
                        "Conversion factor from IA32_QM_CTR to bytes",
                        rmid.conversion_factor(),
                    ),
                    RowGen::tuple("Maximum range of RMID", rmid.maximum_rmid_range()),
                    RowGen::tuple("L3 occupancy monitoring", rmid.has_occupancy_monitoring()),
                    RowGen::tuple(
                        "L3 total bandwidth monitoring",
                        rmid.has_total_bandwidth_monitoring(),
                    ),
                    RowGen::tuple(
                        "L3 local bandwidth monitoring",
                        rmid.has_local_bandwidth_monitoring(),
                    ),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_rdt_allocation_info() {
        print_title(&skin, "Resource Director Technology Allocation (0x10/0)");
        table2(
            &skin,
            &[
                RowGen::tuple("L3 cache allocation technology", info.has_l3_cat()),
                RowGen::tuple("L2 cache allocation technology", info.has_l2_cat()),
                RowGen::tuple(
                    "memory bandwidth allocation",
                    info.has_memory_bandwidth_allocation(),
                ),
            ],
        );

        if let Some(l3_cat) = info.l3_cat() {
            print_subtitle(&skin, "L3 Cache Allocation Technology (0x10/1):");
            table2(
                &skin,
                &[
                    RowGen::tuple("length of capacity bit mask", l3_cat.capacity_mask_length()),
                    RowGen::tuple(
                        "Bit-granular map of isolation/contention",
                        l3_cat.isolation_bitmap(),
                    ),
                    RowGen::tuple(
                        "code and data prioritization",
                        l3_cat.has_code_data_prioritization(),
                    ),
                    RowGen::tuple("highest COS number", l3_cat.highest_cos()),
                ],
            );
        }
        if let Some(l2_cat) = info.l2_cat() {
            print_subtitle(&skin, "L2 Cache Allocation Technology (0x10/2):");
            table2(
                &skin,
                &[
                    RowGen::tuple("length of capacity bit mask", l2_cat.capacity_mask_length()),
                    RowGen::tuple(
                        "Bit-granular map of isolation/contention",
                        l2_cat.isolation_bitmap(),
                    ),
                    RowGen::tuple("highest COS number", l2_cat.highest_cos()),
                ],
            );
        }
        if let Some(mem) = info.memory_bandwidth_allocation() {
            print_subtitle(&skin, "Memory Bandwidth Allocation (0x10/3):");
            table2(
                &skin,
                &[
                    RowGen::tuple("maximum throttling value", mem.max_hba_throttling()),
                    RowGen::tuple("delay values are linear", mem.has_linear_response_delay()),
                    RowGen::tuple("highest COS number", mem.highest_cos()),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_sgx_info() {
        print_title(&skin, "SGX - Software Guard Extensions (0x12/{0,1}):");

        table2(
            &skin,
            &[
                RowGen::tuple("SGX1", info.has_sgx1()),
                RowGen::tuple("SGX2", info.has_sgx2()),
                RowGen::tuple(
                    "SGX ENCLV E*VIRTCHILD, ESETCONTEXT",
                    info.has_enclv_leaves_einvirtchild_edecvirtchild_esetcontext(),
                ),
                RowGen::tuple(
                    "SGX ENCLS ETRACKC, ERDINFO, ELDBC, ELDUC",
                    info.has_encls_leaves_etrackc_erdinfo_eldbc_elduc(),
                ),
                RowGen::tuple("MISCSELECT", info.miscselect()),
                RowGen::tuple(
                    "MaxEnclaveSize_Not64 (log2)",
                    info.max_enclave_size_non_64bit(),
                ),
                RowGen::tuple("MaxEnclaveSize_64 (log2)", info.max_enclave_size_64bit()),
            ],
        );

        for (idx, leaf) in info.iter().enumerate() {
            let SgxSectionInfo::Epc(section) = leaf;
            print_subtitle(
                &skin,
                format!("Enclave Page Cache (0x12/{})", idx + 2).as_str(),
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("physical base address", section.physical_base()),
                    RowGen::tuple("size", section.size()),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_processor_trace_info() {
        print_title(&skin, "Intel Processor Trace (0x14):");
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "IA32_RTIT_CR3_MATCH is accessible",
                    info.has_rtit_cr3_match(),
                ),
                RowGen::tuple(
                    "configurable PSB & cycle-accurate",
                    info.has_configurable_psb_and_cycle_accurate_mode(),
                ),
                RowGen::tuple(
                    "IP & TraceStop filtering; PT preserve",
                    info.has_ip_tracestop_filtering(),
                ),
                RowGen::tuple(
                    "MTC timing packet; suppress COFI-based",
                    info.has_mtc_timing_packet_coefi_suppression(),
                ),
                RowGen::tuple("PTWRITE", info.has_ptwrite()),
                RowGen::tuple("power event trace", info.has_power_event_trace()),
                RowGen::tuple("ToPA output scheme", info.has_topa()),
                RowGen::tuple(
                    "ToPA can hold many output entries",
                    info.has_topa_maximum_entries(),
                ),
                RowGen::tuple(
                    "single-range output scheme support",
                    info.has_single_range_output_scheme(),
                ),
                RowGen::tuple(
                    "output to trace transport",
                    info.has_trace_transport_subsystem(),
                ),
                RowGen::tuple(
                    "IP payloads have LIP values & CS",
                    info.has_lip_with_cs_base(),
                ),
                RowGen::tuple(
                    "configurable address ranges",
                    info.configurable_address_ranges(),
                ),
                RowGen::tuple(
                    "supported MTC periods bitmask",
                    info.supported_mtc_period_encodings(),
                ),
                RowGen::tuple(
                    "supported cycle threshold bitmask",
                    info.supported_cycle_threshold_value_encodings(),
                ),
                RowGen::tuple(
                    "supported config PSB freq bitmask",
                    info.supported_psb_frequency_encodings(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_tsc_info() {
        print_title(
            &skin,
            "Time Stamp Counter/Core Crystal Clock Information (0x15):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "TSC/clock ratio",
                    format!("{} / {}", info.numerator(), info.denominator()),
                ),
                RowGen::tuple("nominal core crystal clock", info.nominal_frequency()),
            ],
        );
    }

    if let Some(info) = cpuid.get_processor_frequency_info() {
        print_title(&skin, "Processor Frequency Information (0x16):");
        table2(
            &skin,
            &[
                RowGen::tuple("Core Base Frequency (MHz)", info.processor_base_frequency()),
                RowGen::tuple(
                    "Core Maximum Frequency (MHz)",
                    info.processor_max_frequency(),
                ),
                RowGen::tuple("Bus (Reference) Frequency (MHz)", info.bus_frequency()),
            ],
        );
    }

    if let Some(dat_iter) = cpuid.get_deterministic_address_translation_info() {
        for (idx, info) in dat_iter.enumerate() {
            print_title(
                &skin,
                format!(
                    "Deterministic Address Translation Structure (0x18/{}):",
                    idx
                )
                .as_str(),
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("number of sets", info.sets()),
                    RowGen::tuple("4 KiB page size entries", info.has_4k_entries()),
                    RowGen::tuple("2 MiB page size entries", info.has_2mb_entries()),
                    RowGen::tuple("4 MiB page size entries", info.has_4mb_entries()),
                    RowGen::tuple("1 GiB page size entries", info.has_1gb_entries()),
                    RowGen::tuple("partitioning", info.partitioning()),
                    RowGen::tuple("ways of associativity", info.ways()),
                    RowGen::tuple("translation cache type", info.cache_type()),
                    RowGen::tuple("translation cache level", info.cache_level()),
                    RowGen::tuple("fully associative", info.is_fully_associative()),
                    RowGen::tuple(
                        "maximum number of addressible IDs",
                        info.max_addressable_ids(),
                    ),
                    RowGen::tuple(
                        "maximum number of addressible IDs",
                        info.max_addressable_ids(),
                    ),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_soc_vendor_info() {
        print_title(&skin, "System-on-Chip (SoC) Vendor Info (0x17):");
        table2(
            &skin,
            &[
                RowGen::tuple("Vendor ID", info.get_soc_vendor_id()),
                RowGen::tuple("Project ID", info.get_project_id()),
                RowGen::tuple("Stepping ID", info.get_stepping_id()),
                RowGen::tuple("Vendor Brand", info.get_vendor_brand()),
            ],
        );

        if let Some(iter) = info.get_vendor_attributes() {
            for (idx, attr) in iter.enumerate() {
                print_cpuid_result(&skin, format!("0x17 {:#x}", idx + 4), attr);
            }
        }
    }

    if let Some(info) = cpuid.get_processor_brand_string() {
        print_attr(
            &skin,
            "Processor Brand String",
            format!("\"**{}**\"", info.as_str()),
        );
    }

    if let Some(info) = cpuid.get_l1_cache_and_tlb_info() {
        print_title(&skin, "L1 TLB 2/4 MiB entries (0x8000_0005/eax):");
        table2(
            &skin,
            &[
                RowGen::tuple("iTLB #entries", info.itlb_2m_4m_size()),
                RowGen::tuple("iTLB associativity", info.itlb_2m_4m_associativity()),
                RowGen::tuple("dTLB #entries", info.dtlb_2m_4m_size()),
                RowGen::tuple("dTLB associativity", info.dtlb_2m_4m_associativity()),
            ],
        );

        print_title(&skin, "L1 TLB 4 KiB entries (0x8000_0005/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple("iTLB #entries", info.itlb_4k_size()),
                RowGen::tuple("iTLB associativity", info.itlb_4k_associativity()),
                RowGen::tuple("dTLB #entries", info.dtlb_4k_size()),
                RowGen::tuple("dTLB associativity", info.dtlb_4k_associativity()),
            ],
        );

        print_title(&skin, "L1 dCache (0x8000_0005/ecx):");
        table2(
            &skin,
            &[
                RowGen::tuple("line size [Bytes]", info.dcache_line_size()),
                RowGen::tuple("lines per tag", info.dcache_lines_per_tag()),
                RowGen::tuple("associativity", info.dcache_associativity()),
                RowGen::tuple("size [KiB]", info.dcache_size()),
            ],
        );

        print_title(&skin, "L1 iCache (0x8000_0005/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("line size [Bytes]", info.icache_line_size()),
                RowGen::tuple("lines per tag", info.icache_lines_per_tag()),
                RowGen::tuple("associativity", info.icache_associativity()),
                RowGen::tuple("size [KiB]", info.icache_size()),
            ],
        );
    }

    if let Some(info) = cpuid.get_l2_l3_cache_and_tlb_info() {
        print_title(&skin, "L2 TLB 2/4 MiB entries (0x8000_0006/eax):");
        table2(
            &skin,
            &[
                RowGen::tuple("iTLB #entries", info.itlb_2m_4m_size()),
                RowGen::tuple("iTLB associativity", info.itlb_2m_4m_associativity()),
                RowGen::tuple("dTLB #entries", info.dtlb_2m_4m_size()),
                RowGen::tuple("dTLB associativity", info.dtlb_2m_4m_associativity()),
            ],
        );

        print_title(&skin, "L2 TLB 4 KiB entries (0x8000_0006/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple("iTLB #entries", info.itlb_4k_size()),
                RowGen::tuple("iTLB associativity", info.itlb_4k_associativity()),
                RowGen::tuple("dTLB #entries", info.dtlb_4k_size()),
                RowGen::tuple("dTLB associativity", info.dtlb_4k_associativity()),
            ],
        );

        print_title(&skin, "L2 Cache (0x8000_0006/ecx):");
        table2(
            &skin,
            &[
                RowGen::tuple("line size [Bytes]", info.l2cache_line_size()),
                RowGen::tuple("lines per tag", info.l2cache_lines_per_tag()),
                RowGen::tuple("associativity", info.l2cache_associativity()),
                RowGen::tuple("size [KiB]", info.l2cache_size()),
            ],
        );

        print_title(&skin, "L3 Cache (0x8000_0006/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("line size [Bytes]", info.l3cache_line_size()),
                RowGen::tuple("lines per tag", info.l3cache_lines_per_tag()),
                RowGen::tuple("associativity", info.l3cache_associativity()),
                RowGen::tuple("size [KiB]", info.l3cache_size() as usize * 512),
            ],
        );
    }

    if let Some(info) = cpuid.get_advanced_power_mgmt_info() {
        print_title(&skin, "RAS Capability (0x8000_0007/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple("MCA overflow recovery", info.has_mca_overflow_recovery()),
                RowGen::tuple("SUCCOR", info.has_succor()),
                RowGen::tuple("HWA: hardware assert", info.has_hwa()),
            ],
        );

        print_title(&skin, "Advanced Power Management (0x8000_0007/ecx):");
        print_attr(
            &skin,
            "Ratio of Compute Unit Power Acc. sample period to TSC",
            info.cpu_pwr_sample_time_ratio(),
        );

        print_title(&skin, "Advanced Power Management (0x8000_0007/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("TS: temperature sensing diode", info.has_ts()),
                RowGen::tuple("FID: frequency ID control", info.has_freq_id_ctrl()),
                RowGen::tuple("VID: voltage ID control", info.has_volt_id_ctrl()),
                RowGen::tuple("TTP: thermal trip", info.has_thermtrip()),
                RowGen::tuple("TM: thermal monitor", info.has_tm()),
                RowGen::tuple("100 MHz multiplier control", info.has_100mhz_steps()),
                RowGen::tuple("hardware P-State control", info.has_hw_pstate()),
                RowGen::tuple("Invariant TSC", info.has_invariant_tsc()),
                RowGen::tuple("CPB: core performance boost", info.has_cpb()),
                RowGen::tuple(
                    "read-only effective frequency interface",
                    info.has_ro_effective_freq_iface(),
                ),
                RowGen::tuple("processor feedback interface", info.has_feedback_iface()),
                RowGen::tuple("APM power reporting", info.has_power_reporting_iface()),
            ],
        );
    }

    if let Some(info) = cpuid.get_processor_capacity_feature_info() {
        print_title(
            &skin,
            "Physical Address and Linear Address Size (0x8000_0008/eax):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "maximum physical address [Bits]",
                    info.physical_address_bits(),
                ),
                RowGen::tuple(
                    "maximum linear (virtual) address [Bits]",
                    info.linear_address_bits(),
                ),
                RowGen::tuple(
                    "maximum guest physical address [Bits]",
                    info.guest_physical_address_bits(),
                ),
            ],
        );

        print_title(&skin, "Extended Feature Extensions ID (0x8000_0008/ebx):");
        table2(
            &skin,
            &[
                RowGen::tuple("CLZERO", info.has_cl_zero()),
                RowGen::tuple("instructions retired count", info.has_inst_ret_cntr_msr()),
                RowGen::tuple(
                    "always save/restore error pointers",
                    info.has_restore_fp_error_ptrs(),
                ),
                RowGen::tuple("RDPRU", info.has_rdpru()),
                RowGen::tuple("INVLPGB", info.has_invlpgb()),
                RowGen::tuple("MCOMMIT", info.has_mcommit()),
                RowGen::tuple("WBNOINVD", info.has_wbnoinvd()),
                RowGen::tuple("WBNOINVD/WBINVD interruptible", info.has_int_wbinvd()),
                RowGen::tuple("EFER.LMSLE unsupported", info.has_unsupported_efer_lmsle()),
                RowGen::tuple("INVLPGB with nested paging", info.has_invlpgb_nested()),
            ],
        );

        print_title(&skin, "Size Identifiers (0x8000_0008/ecx):");
        table2(
            &skin,
            &[
                RowGen::tuple("Logical processors", info.num_phys_threads()),
                RowGen::tuple("APIC core ID size", info.apic_id_size()),
                RowGen::tuple("Max. logical processors", info.maximum_logical_processors()),
                RowGen::tuple("Perf. TSC size [Bits]", info.perf_tsc_size()),
            ],
        );

        print_title(&skin, "Size Identifiers (0x8000_0008/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("RDPRU max. input value", info.max_rdpru_id()),
                RowGen::tuple("INVLPGB max. #pages", info.invlpgb_max_pages()),
            ],
        );
    }

    if let Some(info) = cpuid.get_svm_info() {
        print_title(&skin, "SVM Secure Virtual Machine (0x8000_000a/eax):");
        print_attr(&skin, "Revision", info.revision());

        print_title(&skin, "SVM Secure Virtual Machine (0x8000_000a/edx):");
        table2(
            &skin,
            &[
                RowGen::tuple("nested paging", info.has_nested_paging()),
                RowGen::tuple("LBR virtualization", info.has_lbr_virtualization()),
                RowGen::tuple("SVM lock", info.has_svm_lock()),
                RowGen::tuple("NRIP", info.has_nrip()),
                RowGen::tuple("MSR based TSC rate control", info.has_tsc_rate_msr()),
                RowGen::tuple("VMCB clean bits support", info.has_vmcb_clean_bits()),
                RowGen::tuple("flush by ASID", info.has_flush_by_asid()),
                RowGen::tuple("decode assists", info.has_decode_assists()),
                RowGen::tuple("pause intercept filter", info.has_pause_filter()),
                RowGen::tuple("pause filter threshold", info.has_pause_filter_threshold()),
                RowGen::tuple("AVIC: virtual interrupt controller", info.has_avic()),
                RowGen::tuple(
                    "virtualized VMLOAD/VMSAVE",
                    info.has_vmsave_virtualization(),
                ),
                RowGen::tuple("GIF: virtual global interrupt flag", info.has_gif()),
                RowGen::tuple("GMET: guest mode execute trap", info.has_gmet()),
                RowGen::tuple("SPEC_CTRL virtualization", info.has_spec_ctrl()),
                RowGen::tuple("Supervisor shadow-stack restrictions", info.has_sss_check()),
                RowGen::tuple("#MC intercept", info.has_host_mce_override()),
                RowGen::tuple("INVLPGB/TLBSYNC virtualization", info.has_tlb_ctrl()),
            ],
        );
    }

    if let Some(info) = cpuid.get_tlb_1gb_page_info() {
        print_title(&skin, "TLB 1-GiB Pages Info (0x8000_0019):");
        table2(
            &skin,
            &[
                RowGen::tuple("L1 iTLB #entries", info.itlb_l1_1gb_size()),
                RowGen::tuple("L1 iTLB associativity", info.itlb_l1_1gb_associativity()),
                RowGen::tuple("L1 dTLB #entries", info.dtlb_l1_1gb_size()),
                RowGen::tuple("L1 dTLB associativity", info.dtlb_l1_1gb_associativity()),
                RowGen::tuple("L2 iTLB #entries", info.itlb_l2_1gb_size()),
                RowGen::tuple("L2 iTLB associativity", info.itlb_l2_1gb_associativity()),
                RowGen::tuple("L2 dTLB #entries", info.dtlb_l2_1gb_size()),
                RowGen::tuple("L2 dTLB associativity", info.dtlb_l2_1gb_associativity()),
            ],
        );
    }

    if let Some(info) = cpuid.get_performance_optimization_info() {
        print_title(&skin, "Performance Optimization Info (0x8000_001a):");
        table2(
            &skin,
            &[
                RowGen::tuple("128-bits width the internal FP/SIMD", info.has_fp128()),
                RowGen::tuple(
                    "MOVU SSE are efficient more than MOVL/MOVH",
                    info.has_movu(),
                ),
                RowGen::tuple("256-bits width the internal FP/SIMD", info.has_fp256()),
            ],
        );
    }

    if let Some(info) = cpuid.get_instruction_based_sampling_capabilities() {
        print_title(
            &skin,
            "Instruction-Based Sampling Capabilities (0x8000_001b):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple("IBS feature flags valid", info.has_feature_flags()),
                RowGen::tuple("IBS fetch sampling supporte", info.has_fetch_sampling()),
                RowGen::tuple(
                    "IBS execution sampling supported",
                    info.has_execution_sampling(),
                ),
                RowGen::tuple(
                    "Read write of op counter supported",
                    info.has_read_write_operation_counter(),
                ),
                RowGen::tuple("Op counting mode supported", info.has_operation_counter()),
                RowGen::tuple(
                    "Branch target address reporting supported",
                    info.has_branch_target_address_reporting(),
                ),
                RowGen::tuple(
                    "IbsOpCurCnt and IbsOpMaxCnt extend by 7 bits",
                    info.has_operation_counter_extended(),
                ),
                RowGen::tuple(
                    "Invalid RIP indication supported",
                    info.has_invalid_rip_indication(),
                ),
                RowGen::tuple(
                    "Fused branch micro-op indication supported",
                    info.has_fused_branch_micro_op_indication(),
                ),
                RowGen::tuple(
                    "L3 Miss Filtering for IBS supported",
                    info.has_l3_miss_filtering(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_processor_topology_info() {
        print_title(&skin, "Processor Topology Info (0x8000_001e):");
        table2(
            &skin,
            &[
                RowGen::tuple("x2APIC ID", info.x2apic_id()),
                RowGen::tuple("Core ID", info.core_id()),
                RowGen::tuple("Threads per core", info.threads_per_core()),
                RowGen::tuple("Node ID", info.node_id()),
                RowGen::tuple("Nodes per processor", info.nodes_per_processor()),
            ],
        );
    }

    if let Some(info) = cpuid.get_memory_encryption_info() {
        print_title(&skin, "Memory Encryption Support (0x8000_001f):");
        table2(
            &skin,
            &[
                RowGen::tuple("SME: Secure Memory Encryption", info.has_sme()),
                RowGen::tuple("SEV: Secure Encrypted Virtualization", info.has_sev()),
                RowGen::tuple("Page Flush MSR", info.has_page_flush_msr()),
                RowGen::tuple("SEV-ES: Encrypted State", info.has_sev_es()),
                RowGen::tuple("SEV Secure Nested Paging", info.has_sev_snp()),
                RowGen::tuple("VM Permission Levels", info.has_vmpl()),
                RowGen::tuple(
                    "Hardware cache coherency across encryption domains",
                    info.has_hw_enforced_cache_coh(),
                ),
                RowGen::tuple("SEV guests only with 64-bit host", info.has_64bit_mode()),
                RowGen::tuple("Restricted injection", info.has_restricted_injection()),
                RowGen::tuple("Alternate injection", info.has_alternate_injection()),
                RowGen::tuple(
                    "Full debug state swap for SEV-ES guests",
                    info.has_debug_swap(),
                ),
                RowGen::tuple(
                    "Disallowing IBS use by the host supported",
                    info.has_prevent_host_ibs(),
                ),
                RowGen::tuple("Virtual Transparent Encryption", info.has_vte()),
                RowGen::tuple("C-bit position in page-table", info.c_bit_position()),
                RowGen::tuple(
                    "Physical address bit reduction",
                    info.physical_address_reduction(),
                ),
                RowGen::tuple(
                    "Max. simultaneouslys encrypted guests",
                    info.max_encrypted_guests(),
                ),
                RowGen::tuple(
                    "Minimum ASID value for SEV guest",
                    info.min_sev_no_es_asid(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_pqos_extended_feature_info() {
        print_title(&skin, "Platform Quality of Service (0x8000_0020/0):");
        table2(
            &skin,
            &[
                RowGen::tuple("Memory Bandwidth Enforcement", info.has_l3mbe()),
                RowGen::tuple("Slow Memory Bandwidth Enforcement", info.has_l3smbe()),
                RowGen::tuple("Bandwidth Monitoring Event Configuration", info.has_bmec()),
                RowGen::tuple("L3 Range Reservation", info.has_l3rr()),
                RowGen::tuple("Assignable Bandwidth Monitoring Counters", info.has_abmc()),
                RowGen::tuple(
                    "Smart Data Cache Injection (SDCI) Allocation Enforcement",
                    info.has_sdciae(),
                ),
            ],
        );
        if let Some(info) = info.get_l3_memory_bandwidth_enforcement_info() {
            print_title(
                &skin,
                "L3 Memory Bandwidth Enforcement Information (0x8000_0020/1):",
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("Bandwidth Length", info.bandwidth_length()),
                    RowGen::tuple("COS Number", info.cos_max()),
                ],
            );
        }
        if let Some(info) = info.get_l3_slow_memory_bandwidth_enforcement_info() {
            print_title(
                &skin,
                "L3 Slow Memory Bandwidth Enforcement Information (0x8000_0020/2):",
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("Bandwidth Length", info.bandwidth_length()),
                    RowGen::tuple("COS Number", info.cos_max()),
                ],
            );
        }
        if let Some(info) = info.get_bandwidth_monitoring_event_counters_info() {
            print_title(
                &skin,
                "Bandwidth Monitoring Event Counters Information (0x8000_0020/3):",
            );
            table2(
                &skin,
                &[
                    RowGen::tuple(
                        "Number of configurable bandwidth events",
                        info.number_events(),
                    ),
                    RowGen::tuple(
                        "Reads to local DRAM memory",
                        info.has_l3_cache_lcl_bw_fill_mon(),
                    ),
                    RowGen::tuple(
                        "Reads to remote DRAM memory",
                        info.has_l3_cache_rmt_bw_fill_mon(),
                    ),
                    RowGen::tuple(
                        "Non-temporal writes to local memory",
                        info.has_l3_cache_lcl_bw_nt_wr_mon(),
                    ),
                    RowGen::tuple(
                        "Non-temporal writes to remote memory",
                        info.has_l3_cache_rmt_bw_nt_wr_mon(),
                    ),
                    RowGen::tuple(
                        "Reads to local memory identified as 'Slow Memory'",
                        info.has_l3_cache_lcl_slow_bw_fill_mon(),
                    ),
                    RowGen::tuple(
                        "Reads to remote memory identified as 'Slow Memory'",
                        info.has_l3_cache_rmt_slow_bw_fill_mon(),
                    ),
                    RowGen::tuple(
                        "Dirty victim writes to all types of memory”.",
                        info.has_l3_cache_vic_mon(),
                    ),
                ],
            );
        }
        if let Some(info) = info.get_assignable_bandwidth_monitoring_counters_info() {
            print_title(
                &skin,
                "Assignable Bandwidth Monitoring Counters Information (0x8000_0020/5):",
            );
            table2(
                &skin,
                &[
                    RowGen::tuple(
                        "Indicates that QM_CTR bit 61 is an overflow bit",
                        info.has_overflow_bit(),
                    ),
                    RowGen::tuple(
                        "QM_CTR counter width, offset from 24 bits",
                        info.counter_size(),
                    ),
                    RowGen::tuple("Maximum supported ABMC counter ID", info.max_abmc()),
                    RowGen::tuple(
                        "Bandwidth counters can be configured",
                        info.has_select_cos(),
                    ),
                ],
            );
        }
    }

    if let Some(info) = cpuid.get_extended_feature_identification_2() {
        print_title(&skin, "Extended Feature Identification (0x8000_0021):");
        table2(
            &skin,
            &[
                RowGen::tuple("Processor ignores nested data breakpoints", info.has_no_nested_data_bp()),
                RowGen::tuple("LFENCE is always dispatch serializing", info.has_lfence_always_serializing()),
                RowGen::tuple("SMM paging configuration lock supported", info.has_smm_pg_cfg_lock()),
                RowGen::tuple("Null segment selector loads also clear the destination segment register base and limit", info.has_null_select_clears_base()),
                RowGen::tuple("Upper Address Ignore is supported", info.has_upper_address_ignore()),
                RowGen::tuple("Automatic IBRS", info.has_automatic_ibrs()),
                RowGen::tuple("SMM_CTL MSR (C001_0116h) is not supporte", info.has_no_smm_ctl_msr()),
                RowGen::tuple("Prefetch control MSR supporte", info.has_prefetch_ctl_msr()),
                RowGen::tuple("CPUID disable for non-privileged softwar", info.has_cpuid_user_dis()),
                RowGen::tuple("The size of the Microcode patch in 16-byte multiples", info.microcode_patch_size()),
            ],
        );
    }

    if let Some(info) = cpuid.get_extended_performance_monitoring_and_debug() {
        print_title(
            &skin,
            "Extended Performance Monitoring and Debug (0x8000_0022):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "Performance Monitoring Version 2 supported",
                    info.has_perf_mon_v2(),
                ),
                RowGen::tuple("Last Branch Record Stack supported", info.has_lbr_stack()),
                RowGen::tuple("LbrAndPmcFreeze", info.has_lbr_and_pmc_freeze()),
                RowGen::tuple(
                    "Number of Core Performance Counters",
                    info.num_perf_ctr_core(),
                ),
                RowGen::tuple(
                    "Number of Last Branch Record Stack entries",
                    info.num_lbr_stack_size(),
                ),
                RowGen::tuple(
                    "Number of Northbridge Performance Monitor Counters",
                    info.num_perf_ctr_nb(),
                ),
            ],
        );
    }

    if let Some(info) = cpuid.get_multi_key_encrypted_memory_capabilities() {
        print_title(
            &skin,
            "Multi-Key Encrypted Memory Capabilities (0x8000_0023):",
        );
        table2(
            &skin,
            &[
                RowGen::tuple(
                    "Secure Host Multi-Key Memory (MEM-HMK) Encryption Mode Supported",
                    info.has_mem_hmk(),
                ),
                RowGen::tuple("MaxMemHmkEncrKeyID", info.max_mem_hmk_encr_key_id()),
            ],
        );
    }

    if let Some(info) = cpuid.get_extended_cpu_topology() {
        for (level, info) in info.enumerate() {
            print_title(
                &skin,
                &format!("Extended CPU Topology (0x8000_0026/{level}):"),
            );
            table2(
                &skin,
                &[
                    RowGen::tuple("Mask Width", info.mask_width()),
                    RowGen::tuple(
                        "Efficiency Ranking Available",
                        info.has_efficiency_ranking_available(),
                    ),
                    RowGen::tuple("Heterogeneous Cores", info.has_heterogeneous_cores()),
                    RowGen::tuple("Asymmetric Topology", info.has_asymmetric_topology()),
                    RowGen::tuple(
                        "Number of logical processors at the current hierarchy level",
                        info.num_logical_processors(),
                    ),
                    RowGen::tuple(
                        "Static efficiency ranking between cores of a specific core type",
                        info.pwr_efficiency_ranking(),
                    ),
                    RowGen::tuple("Native Mode Id", info.native_mode_id()),
                    RowGen::tuple("Core Type", info.core_type()),
                    RowGen::tuple("Level Type", info.level_type().to_string()),
                    RowGen::tuple(
                        "Extended APIC ID of the logical processo",
                        info.extended_apic_id(),
                    ),
                ],
            );
        }
    }
}
