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
    unsafe impl ::zerocopy::TryFromBytes for Foo
    where
        u8: ::zerocopy::TryFromBytes,
    {
        fn only_derive_is_allowed_to_implement_this_trait() {}
        #[inline(always)]
        fn is_safe<___ZcAlignment>(
            _candidate: ::zerocopy::Maybe<'_, Self, ___ZcAlignment>,
        ) -> ::zerocopy::util::macro_util::core_reexport::primitive::bool
        where
            ___ZcAlignment: ::zerocopy::invariant::Alignment,
        {
            if false {
                fn assert_is_from_bytes<T>()
                where
                    T: ::zerocopy::FromBytes,
                    T: ?::zerocopy::util::macro_util::core_reexport::marker::Sized,
                {}
                assert_is_from_bytes::<Self>();
            }
            true
        }
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
        enum ẕa {}
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
            unsafe impl ::zerocopy::HasTag<
                ::zerocopy::project_clients::TryFromBytesDerive,
            > for Foo {
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
            unsafe impl ::zerocopy::HasField<
                ::zerocopy::project_clients::TryFromBytesDerive,
                ẕa,
                { ::zerocopy::UNION_VARIANT_ID },
                { ::zerocopy::ident_id!(a) },
            > for Foo {
                fn only_derive_is_allowed_to_implement_this_trait() {}
                type Type = u8;
                #[inline(always)]
                fn project(
                    slf: ::zerocopy::pointer::PtrInner<'_, Self>,
                ) -> *mut <Self as ::zerocopy::HasField<
                    ::zerocopy::project_clients::TryFromBytesDerive,
                    ẕa,
                    { ::zerocopy::UNION_VARIANT_ID },
                    { ::zerocopy::ident_id!(a) },
                >>::Type {
                    let slf = slf.as_ptr();
                    unsafe {
                        ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                            (* slf).a
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
                ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                ___ZcAlignment: ::zerocopy::invariant::Alignment,
            > ::zerocopy::ProjectField<
                ::zerocopy::project_clients::TryFromBytesDerive,
                ẕa,
                (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
                { ::zerocopy::UNION_VARIANT_ID },
                { ::zerocopy::ident_id!(a) },
            > for Foo {
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
                ___ZcAliasing: ::zerocopy::invariant::Aliasing,
                ___ZcAlignment: ::zerocopy::invariant::Alignment,
            > ::zerocopy::ProjectField<
                ::zerocopy::project_clients::TryFromBytesDerive,
                ẕa,
                (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
                { ::zerocopy::UNION_VARIANT_ID },
                { ::zerocopy::ident_id!(a) },
            > for Foo {
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
    unsafe impl ::zerocopy::FromZeros for Foo
    where
        u8: ::zerocopy::FromZeros,
    {
        fn only_derive_is_allowed_to_implement_this_trait() {}
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
    unsafe impl ::zerocopy::FromBytes for Foo
    where
        u8: ::zerocopy::FromBytes,
    {
        fn only_derive_is_allowed_to_implement_this_trait() {}
    }
};
