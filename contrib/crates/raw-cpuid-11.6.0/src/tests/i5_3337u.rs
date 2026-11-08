use crate::*;

#[test]
fn genuine_intel() {
    let vf = VendorInfo {
        ebx: 1970169159,
        edx: 1231384169,
        ecx: 1818588270,
    };
    assert!(vf.as_str() == "GenuineIntel");
}

#[test]
fn feature_info() {
    let finfo = FeatureInfo {
        vendor: Vendor::Intel,
        eax: 198313,
        ebx: 34605056,
        edx_ecx: FeatureInfoFlags::from_bits_truncate(2109399999 | 3219913727 << 32),
    };

    assert!(finfo.base_model_id() == 10);
    assert!(finfo.extended_model_id() == 3);
    assert!(finfo.stepping_id() == 9);
    assert!(finfo.extended_family_id() == 0);
    assert!(finfo.base_family_id() == 6);
    assert!(finfo.stepping_id() == 9);
    assert!(finfo.brand_index() == 0);

    assert!(finfo.edx_ecx.contains(FeatureInfoFlags::SSE2));
    assert!(finfo.edx_ecx.contains(FeatureInfoFlags::SSE41));
}

#[test]
fn cache_info() {
    let cinfos = CacheInfoIter {
        current: 1,
        eax: 1979931137,
        ebx: 15774463,
        ecx: 0,
        edx: 13238272,
    };
    for (idx, cache) in cinfos.enumerate() {
        match idx {
            0 => assert!(cache.num == 0xff),
            1 => assert!(cache.num == 0x5a),
            2 => assert!(cache.num == 0xb2),
            3 => assert!(cache.num == 0x03),
            4 => assert!(cache.num == 0xf0),
            5 => assert!(cache.num == 0xca),
            6 => assert!(cache.num == 0x76),
            _ => unreachable!(),
        }
    }
}

#[test]
fn cache_parameters() {
    let caches: [CacheParameter; 4] = [
        CacheParameter {
            eax: 469778721,
            ebx: 29360191,
            ecx: 63,
            edx: 0,
        },
        CacheParameter {
            eax: 469778722,
            ebx: 29360191,
            ecx: 63,
            edx: 0,
        },
        CacheParameter {
            eax: 469778755,
            ebx: 29360191,
            ecx: 511,
            edx: 0,
        },
        CacheParameter {
            eax: 470008163,
            ebx: 46137407,
            ecx: 4095,
            edx: 6,
        },
    ];

    for (idx, cache) in caches.iter().enumerate() {
        match idx {
            0 => {
                assert!(cache.cache_type() == CacheType::Data);
                assert!(cache.level() == 1);
                assert!(cache.is_self_initializing());
                assert!(!cache.is_fully_associative());
                assert!(cache.max_cores_for_cache() == 2);
                assert!(cache.max_cores_for_package() == 8);
                assert!(cache.coherency_line_size() == 64);
                assert!(cache.physical_line_partitions() == 1);
                assert!(cache.associativity() == 8);
                assert!(!cache.is_write_back_invalidate());
                assert!(!cache.is_inclusive());
                assert!(!cache.has_complex_indexing());
                assert!(cache.sets() == 64);
            }
            1 => {
                assert!(cache.cache_type() == CacheType::Instruction);
                assert!(cache.level() == 1);
                assert!(cache.is_self_initializing());
                assert!(!cache.is_fully_associative());
                assert!(cache.max_cores_for_cache() == 2);
                assert!(cache.max_cores_for_package() == 8);
                assert!(cache.coherency_line_size() == 64);
                assert!(cache.physical_line_partitions() == 1);
                assert!(cache.associativity() == 8);
                assert!(!cache.is_write_back_invalidate());
                assert!(!cache.is_inclusive());
                assert!(!cache.has_complex_indexing());
                assert!(cache.sets() == 64);
            }
            2 => {
                assert!(cache.cache_type() == CacheType::Unified);
                assert!(cache.level() == 2);
                assert!(cache.is_self_initializing());
                assert!(!cache.is_fully_associative());
                assert!(cache.max_cores_for_cache() == 2);
                assert!(cache.max_cores_for_package() == 8);
                assert!(cache.coherency_line_size() == 64);
                assert!(cache.physical_line_partitions() == 1);
                assert!(cache.associativity() == 8);
                assert!(!cache.is_write_back_invalidate());
                assert!(!cache.is_inclusive());
                assert!(!cache.has_complex_indexing());
                assert!(cache.sets() == 512);
            }
            3 => {
                assert!(cache.cache_type() == CacheType::Unified);
                assert!(cache.level() == 3);
                assert!(cache.is_self_initializing());
                assert!(!cache.is_fully_associative());
                assert!(cache.max_cores_for_cache() == 16);
                assert!(cache.max_cores_for_package() == 8);
                assert!(cache.coherency_line_size() == 64);
                assert!(cache.physical_line_partitions() == 1);
                assert!(cache.associativity() == 12);
                assert!(!cache.is_write_back_invalidate());
                assert!(cache.is_inclusive());
                assert!(cache.has_complex_indexing());
                assert!(cache.sets() == 4096);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn monitor_mwait_features() {
    let mmfeatures = MonitorMwaitInfo {
        eax: 64,
        ebx: 64,
        ecx: 3,
        edx: 135456,
    };
    assert!(mmfeatures.smallest_monitor_line() == 64);
    assert!(mmfeatures.largest_monitor_line() == 64);
    assert!(mmfeatures.extensions_supported());
    assert!(mmfeatures.interrupts_as_break_event());
    assert!(mmfeatures.supported_c0_states() == 0);
    assert!(mmfeatures.supported_c1_states() == 2);
    assert!(mmfeatures.supported_c2_states() == 1);
    assert!(mmfeatures.supported_c3_states() == 1);
    assert!(mmfeatures.supported_c4_states() == 2);
    assert!(mmfeatures.supported_c5_states() == 0);
    assert!(mmfeatures.supported_c6_states() == 0);
    assert!(mmfeatures.supported_c7_states() == 0);
}

#[test]
fn thermal_power_features() {
    let tpfeatures = ThermalPowerInfo {
        eax: ThermalPowerFeaturesEax::from_bits_truncate(119),
        ebx: 2,
        ecx: ThermalPowerFeaturesEcx::from_bits_truncate(9),
        _edx: 0,
    };

    assert!(tpfeatures.eax.contains(ThermalPowerFeaturesEax::DTS));
    assert!(tpfeatures
        .eax
        .contains(ThermalPowerFeaturesEax::TURBO_BOOST));
    assert!(tpfeatures.eax.contains(ThermalPowerFeaturesEax::ARAT));
    assert!(tpfeatures.eax.contains(ThermalPowerFeaturesEax::PLN));
    assert!(tpfeatures.eax.contains(ThermalPowerFeaturesEax::ECMD));
    assert!(tpfeatures.eax.contains(ThermalPowerFeaturesEax::PTM));

    assert!(tpfeatures
        .ecx
        .contains(ThermalPowerFeaturesEcx::HW_COORD_FEEDBACK));
    assert!(tpfeatures
        .ecx
        .contains(ThermalPowerFeaturesEcx::ENERGY_BIAS_PREF));

    assert!(tpfeatures.dts_irq_threshold() == 0x2);
    let tpfeatures = ThermalPowerInfo {
        eax: ThermalPowerFeaturesEax::DTS
            | ThermalPowerFeaturesEax::TURBO_BOOST
            | ThermalPowerFeaturesEax::ARAT
            | ThermalPowerFeaturesEax::PLN
            | ThermalPowerFeaturesEax::ECMD
            | ThermalPowerFeaturesEax::PTM
            | ThermalPowerFeaturesEax::HWP
            | ThermalPowerFeaturesEax::HWP_NOTIFICATION
            | ThermalPowerFeaturesEax::HWP_ACTIVITY_WINDOW
            | ThermalPowerFeaturesEax::HWP_ENERGY_PERFORMANCE_PREFERENCE
            | ThermalPowerFeaturesEax::HDC,
        ebx: 2,
        ecx: ThermalPowerFeaturesEcx::HW_COORD_FEEDBACK | ThermalPowerFeaturesEcx::ENERGY_BIAS_PREF,
        _edx: 0,
    };

    assert!(tpfeatures.has_dts());
    assert!(!tpfeatures.has_turbo_boost3());
    assert!(tpfeatures.has_turbo_boost());
    assert!(tpfeatures.has_arat());
    assert!(tpfeatures.has_pln());
    assert!(tpfeatures.has_ecmd());
    assert!(tpfeatures.has_ptm());
    assert!(tpfeatures.has_hwp());
    assert!(tpfeatures.has_hwp_notification());
    assert!(tpfeatures.has_hwp_activity_window());
    assert!(tpfeatures.has_hwp_energy_performance_preference());
    assert!(!tpfeatures.has_hwp_package_level_request());
    assert!(tpfeatures.has_hdc());
    assert!(tpfeatures.has_hw_coord_feedback());
    assert!(tpfeatures.has_energy_bias_pref());
    assert!(tpfeatures.dts_irq_threshold() == 0x2);
}

#[test]
fn extended_features() {
    let tpfeatures = ExtendedFeatures {
        _eax: 0,
        ebx: ExtendedFeaturesEbx::from_bits_truncate(641),
        ecx: ExtendedFeaturesEcx::from_bits_truncate(0),
        edx: ExtendedFeaturesEdx::from_bits_truncate(0),
        eax1: ExtendedFeaturesEax1::from_bits_truncate(0),
        _ebx1: 0,
        _ecx1: 0,
        edx1: ExtendedFeaturesEdx1::from_bits_truncate(0),
    };
    assert!(tpfeatures._eax == 0);
    assert!(tpfeatures.has_fsgsbase());
    assert!(!tpfeatures.has_tsc_adjust_msr());
    assert!(!tpfeatures.has_bmi1());
    assert!(!tpfeatures.has_hle());
    assert!(!tpfeatures.has_avx2());
    assert!(tpfeatures.has_smep());
    assert!(!tpfeatures.has_bmi2());
    assert!(tpfeatures.has_rep_movsb_stosb());
    assert!(!tpfeatures.has_invpcid());
    assert!(!tpfeatures.has_rtm());
    assert!(!tpfeatures.has_rdtm());
    assert!(!tpfeatures.has_fpu_cs_ds_deprecated());

    let tpfeatures2 = ExtendedFeatures {
        _eax: 0,
        ebx: ExtendedFeaturesEbx::FSGSBASE
            | ExtendedFeaturesEbx::ADJUST_MSR
            | ExtendedFeaturesEbx::BMI1
            | ExtendedFeaturesEbx::AVX2
            | ExtendedFeaturesEbx::SMEP
            | ExtendedFeaturesEbx::BMI2
            | ExtendedFeaturesEbx::REP_MOVSB_STOSB
            | ExtendedFeaturesEbx::INVPCID
            | ExtendedFeaturesEbx::DEPRECATE_FPU_CS_DS
            | ExtendedFeaturesEbx::MPX
            | ExtendedFeaturesEbx::RDSEED
            | ExtendedFeaturesEbx::ADX
            | ExtendedFeaturesEbx::SMAP
            | ExtendedFeaturesEbx::CLFLUSHOPT
            | ExtendedFeaturesEbx::PROCESSOR_TRACE,
        ecx: ExtendedFeaturesEcx::from_bits_truncate(0),
        edx: ExtendedFeaturesEdx::from_bits_truncate(201326592),
        eax1: ExtendedFeaturesEax1::from_bits_truncate(0),
        _ebx1: 0,
        _ecx1: 0,
        edx1: ExtendedFeaturesEdx1::from_bits_truncate(0),
    };

    assert!(tpfeatures2.has_fsgsbase());
    assert!(tpfeatures2.has_tsc_adjust_msr());
    assert!(tpfeatures2.has_bmi1());
    assert!(tpfeatures2.has_avx2());
    assert!(tpfeatures2.has_smep());
    assert!(tpfeatures2.has_bmi2());
    assert!(tpfeatures2.has_rep_movsb_stosb());
    assert!(tpfeatures2.has_invpcid());
    assert!(tpfeatures2.has_fpu_cs_ds_deprecated());
    assert!(tpfeatures2.has_mpx());
    assert!(tpfeatures2.has_rdseed());
    assert!(tpfeatures2.has_adx());
    assert!(tpfeatures2.has_smap());
    assert!(tpfeatures2.has_clflushopt());
    assert!(tpfeatures2.has_processor_trace());

    assert!(!tpfeatures2.has_avx512_4vnniw());
    assert!(!tpfeatures2.has_avx512_4fmaps());
    assert!(!tpfeatures2.has_avx512_vp2intersect());
    assert!(!tpfeatures2.has_amx_bf16());
    assert!(!tpfeatures2.has_avx512_fp16());
    assert!(!tpfeatures2.has_amx_tile());
    assert!(!tpfeatures2.has_amx_int8());

    assert!(!tpfeatures2.has_avx_vnni());
    assert!(!tpfeatures2.has_avx512_bf16());
    assert!(!tpfeatures2.has_fzrm());
    assert!(!tpfeatures2.has_fsrs());
    assert!(!tpfeatures2.has_fsrcrs());
    assert!(!tpfeatures2.has_hreset());
    assert!(!tpfeatures2.has_avx_ifma());
    assert!(!tpfeatures2.has_lam());
    assert!(!tpfeatures2.has_msrlist());
    assert!(!tpfeatures2.has_invd_disable_post_bios_done());
    assert!(!tpfeatures2.has_avx_vnni_int8());
    assert!(!tpfeatures2.has_avx_ne_convert());
    assert!(!tpfeatures2.has_avx_vnni_int16());
    assert!(!tpfeatures2.has_prefetchi());
    assert!(!tpfeatures2.has_uiret_uif());
    assert!(!tpfeatures2.has_cet_sss());
    assert!(!tpfeatures2.has_avx10());
}

#[test]
fn direct_cache_access_info() {
    let dca = DirectCacheAccessInfo { eax: 0x1 };
    assert!(dca.get_dca_cap_value() == 0x1);
}

#[test]
fn performance_monitoring_info() {
    let pm = PerformanceMonitoringInfo {
        eax: 120587267,
        ebx: PerformanceMonitoringFeaturesEbx::from_bits_truncate(0),
        _ecx: 0,
        edx: 1539,
    };

    assert!(pm.version_id() == 3);
    assert!(pm.number_of_counters() == 4);
    assert!(pm.counter_bit_width() == 48);
    assert!(pm.ebx_length() == 7);
    assert!(pm.fixed_function_counters() == 3);
    assert!(pm.fixed_function_counters_bit_width() == 48);

    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::CORE_CYC_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::INST_RET_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::REF_CYC_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::CACHE_REF_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::LL_CACHE_MISS_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::BRANCH_INST_RET_EV_UNAVAILABLE));
    assert!(!pm
        .ebx
        .contains(PerformanceMonitoringFeaturesEbx::BRANCH_MISPRED_EV_UNAVAILABLE));
}

#[cfg(test)]
#[test]
fn extended_topology_info() {
    let l1 = ExtendedTopologyLevel {
        eax: 1,
        ebx: 2,
        ecx: 256,
        edx: 3,
    };
    let l2 = ExtendedTopologyLevel {
        eax: 4,
        ebx: 4,
        ecx: 513,
        edx: 3,
    };

    assert!(l1.processors() == 2);
    assert!(l1.level_number() == 0);
    assert!(l1.level_type() == TopologyType::SMT);
    assert!(l1.x2apic_id() == 3);
    assert!(l1.shift_right_for_next_apic_id() == 1);

    assert!(l2.processors() == 4);
    assert!(l2.level_number() == 1);
    assert!(l2.level_type() == TopologyType::Core);
    assert!(l2.x2apic_id() == 3);
    assert!(l2.shift_right_for_next_apic_id() == 4);
}

#[cfg(test)]
#[test]
fn extended_topology_info_v2() {
    let l1 = ExtendedTopologyLevel {
        eax: 1,
        ebx: 2,
        ecx: 256,
        edx: 3,
    };
    let l2 = ExtendedTopologyLevel {
        eax: 4,
        ebx: 4,
        ecx: 513,
        edx: 3,
    };

    assert!(l1.processors() == 2);
    assert!(l1.level_number() == 0);
    assert!(l1.level_type() == TopologyType::SMT);
    assert!(l1.x2apic_id() == 3);
    assert!(l1.shift_right_for_next_apic_id() == 1);

    assert!(l2.processors() == 4);
    assert!(l2.level_number() == 1);
    assert!(l2.level_type() == TopologyType::Core);
    assert!(l2.x2apic_id() == 3);
    assert!(l2.shift_right_for_next_apic_id() == 4);
}

#[test]
fn extended_state_info() {
    let es = ExtendedStateInfo {
        read: CpuIdReaderNative,
        eax: ExtendedStateInfoXCR0Flags::from_bits_truncate(7),
        ebx: 832,
        ecx: 832,
        _edx: 0,
        eax1: 1,
        ebx1: 0,
        ecx1: ExtendedStateInfoXSSFlags::from_bits_truncate(0),
        _edx1: 0,
    };

    assert!(es.xsave_area_size_enabled_features() == 832);
    assert!(es.xsave_area_size_supported_features() == 832);
    assert!(es.has_xsaveopt());
}

#[test]
fn extended_state_info3() {
    /*let cpuid = CpuId::new();
    cpuid.get_extended_state_info().map(|info| {
        println!("{:?}", info);
        use std::vec::Vec;
        let es: Vec<ExtendedState> = info.iter().collect();
        println!("{:?}", es);
    });*/

    let esi = ExtendedStateInfo {
        read: CpuIdReaderNative,
        eax: ExtendedStateInfoXCR0Flags::LEGACY_X87
            | ExtendedStateInfoXCR0Flags::SSE128
            | ExtendedStateInfoXCR0Flags::AVX256
            | ExtendedStateInfoXCR0Flags::MPX_BNDREGS
            | ExtendedStateInfoXCR0Flags::MPX_BNDCSR
            | ExtendedStateInfoXCR0Flags::AVX512_OPMASK
            | ExtendedStateInfoXCR0Flags::AVX512_ZMM_HI256
            | ExtendedStateInfoXCR0Flags::AVX512_ZMM_HI16
            | ExtendedStateInfoXCR0Flags::PKRU,
        ebx: 2688,
        ecx: 2696,
        _edx: 0,
        eax1: 15,
        ebx1: 2560,
        ecx1: ExtendedStateInfoXSSFlags::PT,
        _edx1: 0,
    };

    assert!(esi.xcr0_supports_legacy_x87());
    assert!(esi.xcr0_supports_sse_128());
    assert!(esi.xcr0_supports_avx_256());
    assert!(esi.xcr0_supports_mpx_bndregs());
    assert!(esi.xcr0_supports_mpx_bndcsr());
    assert!(esi.xcr0_supports_avx512_opmask());
    assert!(esi.xcr0_supports_avx512_zmm_hi256());
    assert!(esi.xcr0_supports_avx512_zmm_hi16());
    assert!(esi.xcr0_supports_pkru());
    assert!(esi.ia32_xss_supports_pt());
    assert!(!esi.ia32_xss_supports_hdc());

    assert!(esi.xsave_area_size_enabled_features() == 2688);
    assert!(esi.xsave_area_size_supported_features() == 2696);

    assert!(esi.has_xsaveopt());
    assert!(esi.has_xsavec());
    assert!(esi.has_xgetbv());
    assert!(esi.has_xsaves_xrstors());
    assert!(esi.xsave_size() == 2560);

    let es = [
        ExtendedState {
            subleaf: 2,
            eax: 256,
            ebx: 576,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 3,
            eax: 64,
            ebx: 960,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 4,
            eax: 64,
            ebx: 1024,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 5,
            eax: 64,
            ebx: 1088,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 6,
            eax: 512,
            ebx: 1152,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 7,
            eax: 1024,
            ebx: 1664,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 8,
            eax: 128,
            ebx: 0,
            ecx: 1,
        },
        ExtendedState {
            subleaf: 9,
            eax: 8,
            ebx: 2688,
            ecx: 0,
        },
    ];

    let e = &es[0];
    assert!(e.subleaf == 2);
    assert!(e.size() == 256);
    assert!(e.offset() == 576);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[1];
    assert!(e.subleaf == 3);
    assert!(e.size() == 64);
    assert!(e.offset() == 960);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[2];
    assert!(e.subleaf == 4);
    assert!(e.size() == 64);
    assert!(e.offset() == 1024);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[3];
    assert!(e.subleaf == 5);
    assert!(e.size() == 64);
    assert!(e.offset() == 1088);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[4];
    assert!(e.subleaf == 6);
    assert!(e.size() == 512);
    assert!(e.offset() == 1152);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[5];
    assert!(e.subleaf == 7);
    assert!(e.size() == 1024);
    assert!(e.offset() == 1664);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[6];
    assert!(e.subleaf == 8);
    assert!(e.size() == 128);
    assert!(e.offset() == 0);
    assert!(!e.is_in_xcr0());
    assert!(e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());

    let e = &es[7];
    assert!(e.subleaf == 9);
    assert!(e.size() == 8);
    assert!(e.offset() == 2688);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
    assert!(!e.is_compacted_format());
}

#[test]
fn extended_state_info2() {
    let es = ExtendedStateInfo {
        read: CpuIdReaderNative,
        eax: ExtendedStateInfoXCR0Flags::from_bits_truncate(31),
        ebx: 1088,
        ecx: 1088,
        _edx: 0,
        eax1: 15,
        ebx1: 960,
        ecx1: ExtendedStateInfoXSSFlags::from_bits_truncate(256),
        _edx1: 0,
    };

    assert!(es.xcr0_supports_legacy_x87());
    assert!(es.xcr0_supports_sse_128());
    assert!(es.xcr0_supports_avx_256());
    assert!(es.xcr0_supports_mpx_bndregs());
    assert!(es.xcr0_supports_mpx_bndcsr());
    assert!(!es.xcr0_supports_avx512_opmask());
    assert!(!es.xcr0_supports_pkru());
    assert!(es.ia32_xss_supports_pt());

    assert!(es.xsave_area_size_enabled_features() == 0x440);
    assert!(es.xsave_area_size_supported_features() == 0x440);

    assert!(es.has_xsaveopt());
    assert!(es.has_xsavec());
    assert!(es.has_xgetbv());
    assert!(es.has_xsaves_xrstors());
    assert!(es.xsave_size() == 0x3c0);

    let esiter: [ExtendedState; 3] = [
        ExtendedState {
            subleaf: 2,
            eax: 256,
            ebx: 576,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 3,
            eax: 64,
            ebx: 960,
            ecx: 0,
        },
        ExtendedState {
            subleaf: 4,
            eax: 64,
            ebx: 1024,
            ecx: 0,
        },
    ];

    let e = &esiter[0];
    assert!(e.subleaf == 2);
    assert!(e.size() == 256);
    assert!(e.offset() == 576);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());

    let e = &esiter[1];
    assert!(e.subleaf == 3);
    assert!(e.size() == 64);
    assert!(e.offset() == 960);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());

    let e = &esiter[2];
    assert!(e.subleaf == 4);
    assert!(e.size() == 64);
    assert!(e.offset() == 1024);
    assert!(e.is_in_xcr0());
    assert!(!e.is_in_ia32_xss());
}

#[test]
fn quality_of_service_info() {
    let qos = RdtMonitoringInfo {
        read: CpuIdReaderNative,
        ebx: 832,
        edx: 0,
    };

    assert!(qos.rmid_range() == 832);
    assert!(!qos.has_l3_monitoring());
}

#[test]
fn extended_processor_feature_identifiers() {
    let ef = ExtendedProcessorFeatureIdentifiers::new(
        Vendor::Intel,
        CpuIdResult {
            eax: 0,
            ebx: 0,
            ecx: 1,
            edx: 672139264,
        },
    );

    assert!(ef.has_lahf_sahf());
    assert!(!ef.has_lzcnt());
    assert!(!ef.has_prefetchw());
    assert!(ef.has_syscall_sysret());
    assert!(ef.has_execute_disable());
    assert!(!ef.has_1gib_pages());
    assert!(ef.has_rdtscp());
    assert!(ef.has_64bit_mode());
}

#[test]
fn processor_brand_string() {
    let pbs = crate::extended::ProcessorBrandString::new([
        CpuIdResult {
            eax: 538976288,
            ebx: 1226842144,
            ecx: 1818588270,
            edx: 539578920,
        },
        CpuIdResult {
            eax: 1701998403,
            ebx: 692933672,
            ecx: 758475040,
            edx: 926102323,
        },
        CpuIdResult {
            eax: 1346576469,
            ebx: 541073493,
            ecx: 808988209,
            edx: 8013895,
        },
    ]);

    assert_eq!(pbs.as_str(), "Intel(R) Core(TM) i5-3337U CPU @ 1.80GHz");
}

#[cfg(test)]
#[test]
fn sgx_test() {
    let sgx = SgxInfo {
        read: CpuIdReaderNative,
        eax: 1,
        ebx: 0,
        _ecx: 0,
        edx: 9247,
        eax1: 54,
        ebx1: 0,
        ecx1: 31,
        edx1: 0,
    };

    assert!(sgx.max_enclave_size_64bit() == 0x24);
    assert!(sgx.max_enclave_size_non_64bit() == 0x1f);
    assert!(sgx.has_sgx1());
    assert!(!sgx.has_sgx2());
    assert!(sgx.miscselect() == 0x0);
    assert!(sgx.secs_attributes() == (0x0000000000000036, 0x000000000000001f));
}
