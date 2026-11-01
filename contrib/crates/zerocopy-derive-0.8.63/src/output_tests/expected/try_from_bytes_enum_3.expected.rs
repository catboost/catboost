#[allow(
    deprecated,
    private_bounds,
    non_local_definitions,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    non_ascii_idents,
    clippy::missing_inline_in_public_items,
)]
#[deny(ambiguous_associated_items)]
#[automatically_derived]
const _: () = {
    unsafe impl<'a: 'static, X, Y: Deref, const N: usize> ::zerocopy::TryFromBytes
    for ComplexWithGenerics<'a, { N }, X, Y>
    where
        X: Deref<Target = &'a [(X, Y); N]>,
        u8: ::zerocopy::TryFromBytes,
        X: ::zerocopy::TryFromBytes,
        X::Target: ::zerocopy::TryFromBytes,
        Y::Target: ::zerocopy::TryFromBytes,
        [(X, Y); N]: ::zerocopy::TryFromBytes,
        bool: ::zerocopy::TryFromBytes,
        Y: ::zerocopy::TryFromBytes,
        PhantomData<&'a [(X, Y); N]>: ::zerocopy::TryFromBytes,
    {
        fn only_derive_is_allowed_to_implement_this_trait() {}
        #[inline]
        fn is_safe<___ZcAlignment>(
            mut candidate: ::zerocopy::Maybe<'_, Self, ___ZcAlignment>,
        ) -> ::zerocopy::util::macro_util::core_reexport::primitive::bool
        where
            ___ZcAlignment: ::zerocopy::invariant::Alignment,
        {
            #[repr(C)]
            #[allow(dead_code, clippy::derive_partial_eq_without_eq)]
            #[derive(Copy, Clone, PartialEq)]
            pub enum ___ZerocopyTag {
                UnitLike,
                StructLike,
                TupleLike,
            }
            unsafe impl ::zerocopy::Immutable for ___ZerocopyTag {
                fn only_derive_is_allowed_to_implement_this_trait() {}
            }
            type ___ZerocopyTagPrimitive = ::zerocopy::util::macro_util::SizeToTag<
                {
                    ::zerocopy::util::macro_util::core_reexport::mem::size_of::<
                        ___ZerocopyTag,
                    >()
                },
            >;
            const ___ZEROCOPY_TAG_UnitLike: ___ZerocopyTagPrimitive = ___ZerocopyTag::UnitLike
                as ___ZerocopyTagPrimitive;
            const ___ZEROCOPY_TAG_StructLike: ___ZerocopyTagPrimitive = ___ZerocopyTag::StructLike
                as ___ZerocopyTagPrimitive;
            const ___ZEROCOPY_TAG_TupleLike: ___ZerocopyTagPrimitive = ___ZerocopyTag::TupleLike
                as ___ZerocopyTagPrimitive;
            type ___ZerocopyOuterTag = ___ZerocopyTag;
            type ___ZerocopyInnerTag = ();
            #[repr(C)]
            struct ___ZerocopyVariantStruct_StructLike<
                'a: 'static,
                const N: usize,
                X,
                Y: Deref,
            >(
                ::zerocopy::util::macro_util::core_reexport::mem::MaybeUninit<
                    ___ZerocopyInnerTag,
                >,
                u8,
                X,
                X::Target,
                Y::Target,
                [(X, Y); N],
                ::zerocopy::util::macro_util::core_reexport::marker::PhantomData<
                    ComplexWithGenerics<'a, N, X, Y>,
                >,
            )
            where
                X: Deref<Target = &'a [(X, Y); N]>;
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                enum ẕ0 {}
                enum ẕ1 {}
                enum ẕ2 {}
                enum ẕ3 {}
                enum ẕ4 {}
                enum ẕ5 {}
                enum ẕ6 {}
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasTag<::zerocopy::project_clients::TryFromBytesDerive>
                    for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Tag = ();
                        type ProjectToTag = ::zerocopy::pointer::cast::CastToUnit;
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::mem::MaybeUninit<
                            ___ZerocopyInnerTag,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ0,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(0) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).0
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = u8;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ1,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(1) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).1
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = X;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ2,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(2) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).2
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = X::Target;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ3,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(3) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).3
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = Y::Target;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ4,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(4) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).4
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ5,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(5) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = [(X, Y); N];
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ5,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(5) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).5
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ5,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(5) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ5,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(5) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ5,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(5) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ6,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(6) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::marker::PhantomData<
                            ComplexWithGenerics<'a, N, X, Y>,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ6,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(6) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).6
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ6,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(6) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ6,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(6) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ6,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(6) },
                    > for ___ZerocopyVariantStruct_StructLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
            };
            #[repr(C)]
            struct ___ZerocopyVariantStruct_TupleLike<
                'a: 'static,
                const N: usize,
                X,
                Y: Deref,
            >(
                ::zerocopy::util::macro_util::core_reexport::mem::MaybeUninit<
                    ___ZerocopyInnerTag,
                >,
                bool,
                Y,
                PhantomData<&'a [(X, Y); N]>,
                ::zerocopy::util::macro_util::core_reexport::marker::PhantomData<
                    ComplexWithGenerics<'a, N, X, Y>,
                >,
            )
            where
                X: Deref<Target = &'a [(X, Y); N]>;
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                enum ẕ0 {}
                enum ẕ1 {}
                enum ẕ2 {}
                enum ẕ3 {}
                enum ẕ4 {}
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasTag<::zerocopy::project_clients::TryFromBytesDerive>
                    for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Tag = ();
                        type ProjectToTag = ::zerocopy::pointer::cast::CastToUnit;
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::mem::MaybeUninit<
                            ___ZerocopyInnerTag,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ0,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(0) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).0
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ0,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(0) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = bool;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ1,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(1) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).1
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ1,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(1) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = Y;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ2,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(2) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).2
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ2,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(2) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = PhantomData<&'a [(X, Y); N]>;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ3,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(3) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).3
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ3,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(3) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::marker::PhantomData<
                            ComplexWithGenerics<'a, N, X, Y>,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ4,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(4) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).4
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ4,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(4) },
                    > for ___ZerocopyVariantStruct_TupleLike<'a, { N }, X, Y>
                    where
                        X: Deref<Target = &'a [(X, Y); N]>,
                    {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
            };
            #[repr(C)]
            union ___ZerocopyVariants<'a: 'static, const N: usize, X, Y: Deref> {
                __field_StructLike: ::zerocopy::util::macro_util::core_reexport::mem::ManuallyDrop<
                    ___ZerocopyVariantStruct_StructLike<'a, N, X, Y>,
                >,
                __field_TupleLike: ::zerocopy::util::macro_util::core_reexport::mem::ManuallyDrop<
                    ___ZerocopyVariantStruct_TupleLike<'a, N, X, Y>,
                >,
                __nonempty: (),
            }
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                enum ẕ__field_StructLike {}
                enum ẕ__field_TupleLike {}
                enum ẕ__nonempty {}
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasTag<::zerocopy::project_clients::TryFromBytesDerive>
                    for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Tag = ();
                        type ProjectToTag = ::zerocopy::pointer::cast::CastToUnit;
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_StructLike,
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_StructLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::mem::ManuallyDrop<
                            ___ZerocopyVariantStruct_StructLike<'a, N, X, Y>,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ__field_StructLike,
                            { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                            { ::zerocopy::ident_id!(__field_StructLike) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).__field_StructLike
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_StructLike,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_StructLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_StructLike,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_StructLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_TupleLike,
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_TupleLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ::zerocopy::util::macro_util::core_reexport::mem::ManuallyDrop<
                            ___ZerocopyVariantStruct_TupleLike<'a, N, X, Y>,
                        >;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ__field_TupleLike,
                            { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                            { ::zerocopy::ident_id!(__field_TupleLike) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).__field_TupleLike
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_TupleLike,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_TupleLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__field_TupleLike,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__field_TupleLike) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__nonempty,
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__nonempty) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ();
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕ__nonempty,
                            { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                            { ::zerocopy::ident_id!(__nonempty) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).__nonempty
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__nonempty,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__nonempty) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕ__nonempty,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                        { ::zerocopy::ident_id!(__nonempty) },
                    > for ___ZerocopyVariants<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
            };
            #[repr(C)]
            struct ___ZerocopyRawEnum<'a: 'static, const N: usize, X, Y: Deref> {
                tag: ___ZerocopyOuterTag,
                variants: ___ZerocopyVariants<'a, N, X, Y>,
            }
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                enum ẕtag {}
                enum ẕvariants {}
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasTag<::zerocopy::project_clients::TryFromBytesDerive>
                    for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Tag = ();
                        type ProjectToTag = ::zerocopy::pointer::cast::CastToUnit;
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕtag,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(tag) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ___ZerocopyOuterTag;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕtag,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(tag) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).tag
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕtag,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(tag) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕtag,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(tag) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕtag,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(tag) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        const N: usize,
                    > ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕvariants,
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(variants) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Type = ___ZerocopyVariants<'a, N, X, Y>;
                        #[inline(always)]
                        fn project(
                            slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                        ) -> *mut <Self as ::zerocopy::HasField<
                            ::zerocopy::project_clients::TryFromBytesDerive,
                            ẕvariants,
                            { ::zerocopy::STRUCT_VARIANT_ID },
                            { ::zerocopy::ident_id!(variants) },
                        >>::Type {
                            let slf = slf.as_ptr();
                            unsafe {
                                ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                                    (* slf).variants
                                )
                            }
                        }
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕvariants,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(variants) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Uninit,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕvariants,
                        (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        ),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(variants) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Initialized,
                        );
                    }
                };
                #[allow(
                    deprecated,
                    private_bounds,
                    non_local_definitions,
                    non_camel_case_types,
                    non_upper_case_globals,
                    non_snake_case,
                    non_ascii_idents,
                    clippy::missing_inline_in_public_items,
                )]
                #[deny(ambiguous_associated_items)]
                #[automatically_derived]
                const _: () = {
                    unsafe impl<
                        'a: 'static,
                        X,
                        Y: Deref,
                        ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                        ___ZcAlignment: ::zerocopy::invariant::Alignment,
                        const N: usize,
                    > ::zerocopy::ProjectField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        ẕvariants,
                        (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                        { ::zerocopy::STRUCT_VARIANT_ID },
                        { ::zerocopy::ident_id!(variants) },
                    > for ___ZerocopyRawEnum<'a, { N }, X, Y> {
                        fn only_derive_is_allowed_to_implement_this_trait() {}
                        type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                        type Invariants = (
                            ___ZcAliasing,
                            ___ZcAlignment,
                            ::zerocopy::invariant::Safe,
                        );
                    }
                };
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasTag<::zerocopy::project_clients::TryFromBytesDerive>
                for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Tag = ___ZerocopyTag;
                    type ProjectToTag = ::zerocopy::pointer::cast::CastSized;
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(a) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = u8;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(StructLike) },
                        { ::zerocopy::ident_id!(a) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_StructLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(1) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(a) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(a) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(a) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::StructLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(b) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = X;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(StructLike) },
                        { ::zerocopy::ident_id!(b) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_StructLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(2) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(b) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(b) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(b) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::StructLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(c) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = X::Target;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(StructLike) },
                        { ::zerocopy::ident_id!(c) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_StructLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(3) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(c) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(c) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(c) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::StructLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(d) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = Y::Target;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(StructLike) },
                        { ::zerocopy::ident_id!(d) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_StructLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(4) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(d) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(d) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(d) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::StructLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(e) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = [(X, Y); N];
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(StructLike) },
                        { ::zerocopy::ident_id!(e) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_StructLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(5) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(e) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(e) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(StructLike) },
                    { ::zerocopy::ident_id!(e) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::StructLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(0) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = bool;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(TupleLike) },
                        { ::zerocopy::ident_id!(0) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_TupleLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(1) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(0) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(0) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(0) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::TupleLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(1) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = Y;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(TupleLike) },
                        { ::zerocopy::ident_id!(1) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_TupleLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(2) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(1) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(1) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(1) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::TupleLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    const N: usize,
                > ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(2) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Type = PhantomData<&'a [(X, Y); N]>;
                    #[inline(always)]
                    fn project(
                        slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                    ) -> *mut <Self as ::zerocopy::HasField<
                        ::zerocopy::project_clients::TryFromBytesDerive,
                        (),
                        { ::zerocopy::ident_id!(TupleLike) },
                        { ::zerocopy::ident_id!(2) },
                    >>::Type {
                        use ::zerocopy::pointer::cast::{CastSized, Projection};
                        slf.project::<___ZerocopyRawEnum<'a, N, X, Y>, CastSized>()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(variants) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::REPR_C_UNION_VARIANT_ID },
                                    { ::zerocopy::ident_id!(__field_TupleLike) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(value) },
                                >,
                            >()
                            .project::<
                                _,
                                Projection<
                                    ::zerocopy::project_clients::TryFromBytesDerive,
                                    _,
                                    { ::zerocopy::STRUCT_VARIANT_ID },
                                    { ::zerocopy::ident_id!(3) },
                                >,
                            >()
                            .as_ptr()
                    }
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(2) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Uninit,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(2) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ::zerocopy::util::macro_util::core_reexport::convert::Infallible;
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Initialized,
                    );
                }
            };
            #[allow(
                deprecated,
                private_bounds,
                non_local_definitions,
                non_camel_case_types,
                non_upper_case_globals,
                non_snake_case,
                non_ascii_idents,
                clippy::missing_inline_in_public_items,
            )]
            #[deny(ambiguous_associated_items)]
            #[automatically_derived]
            const _: () = {
                unsafe impl<
                    'a: 'static,
                    X,
                    Y: Deref,
                    ___ZcAliasing: ::zerocopy::invariant::Reference,
                    ___ZcAlignment: ::zerocopy::invariant::Alignment,
                    const N: usize,
                > ::zerocopy::ProjectField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    (),
                    (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
                    { ::zerocopy::ident_id!(TupleLike) },
                    { ::zerocopy::ident_id!(2) },
                > for ComplexWithGenerics<'a, { N }, X, Y>
                where
                    X: Deref<Target = &'a [(X, Y); N]>,
                {
                    fn only_derive_is_allowed_to_implement_this_trait() {}
                    type Error = ();
                    type Invariants = (
                        ___ZcAliasing,
                        ___ZcAlignment,
                        ::zerocopy::invariant::Safe,
                    );
                    #[inline(always)]
                    fn is_projectable(
                        tag: ::zerocopy::pointer::Ptr<
                            '_,
                            <Self as ::zerocopy::HasTag<
                                ::zerocopy::project_clients::TryFromBytesDerive,
                            >>::Tag,
                            (
                                ::zerocopy::invariant::Shared,
                                ___ZcAlignment,
                                ::zerocopy::invariant::Safe,
                            ),
                        >,
                    ) -> ::zerocopy::util::macro_util::core_reexport::result::Result<
                        (),
                        (),
                    > {
                        let tag = tag.read::<::zerocopy::BecauseImmutable>();
                        if tag == ___ZerocopyTag::TupleLike {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Ok(())
                        } else {
                            ::zerocopy::util::macro_util::core_reexport::result::Result::Err(())
                        }
                    }
                }
            };
            let tag = candidate
                .reborrow()
                .cast::<
                    ___ZerocopyTagPrimitive,
                    ::zerocopy::pointer::cast::CastSized,
                    (::zerocopy::pointer::BecauseRead, _),
                >()
                .recall_validity::<_, (_, (_, _))>()
                .read::<::zerocopy::BecauseImmutable>();
            match tag {
                ___ZEROCOPY_TAG_UnitLike => true,
                ___ZEROCOPY_TAG_StructLike => {
                    true
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(StructLike) }, {
                            ::zerocopy::ident_id!(a) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(StructLike) }, {
                            ::zerocopy::ident_id!(b) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(StructLike) }, {
                            ::zerocopy::ident_id!(c) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(StructLike) }, {
                            ::zerocopy::ident_id!(d) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(StructLike) }, {
                            ::zerocopy::ident_id!(e) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                }
                ___ZEROCOPY_TAG_TupleLike => {
                    true
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(TupleLike) }, {
                            ::zerocopy::ident_id!(0) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(TupleLike) }, {
                            ::zerocopy::ident_id!(1) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                        && ::zerocopy::into_inner!(
                            candidate.project:: <
                            ::zerocopy::project_clients::TryFromBytesDerive, _, {
                            ::zerocopy::ident_id!(TupleLike) }, {
                            ::zerocopy::ident_id!(2) }, > ()
                        )
                            .try_into_safe::<_, ::zerocopy::BecauseImmutable>()
                            .is_ok()
                }
                _ => false,
            }
        }
    }
};
