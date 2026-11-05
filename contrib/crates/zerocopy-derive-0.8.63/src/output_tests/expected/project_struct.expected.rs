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
    enum ẕfield {}
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
        unsafe impl ::zerocopy::HasTag<::zerocopy::project_clients::ProjectDerive>
        for Foo {
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
            ::zerocopy::project_clients::ProjectDerive,
            ẕfield,
            { ::zerocopy::STRUCT_VARIANT_ID },
            { ::zerocopy::ident_id!(field) },
        > for Foo {
            fn only_derive_is_allowed_to_implement_this_trait() {}
            type Type = u8;
            #[inline(always)]
            fn project(
                slf: ::zerocopy::pointer::PtrInner<'_, Self>,
            ) -> *mut <Self as ::zerocopy::HasField<
                ::zerocopy::project_clients::ProjectDerive,
                ẕfield,
                { ::zerocopy::STRUCT_VARIANT_ID },
                { ::zerocopy::ident_id!(field) },
            >>::Type {
                let slf = slf.as_ptr();
                unsafe {
                    ::zerocopy::util::macro_util::core_reexport::ptr::addr_of_mut!(
                        (* slf).field
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
            ::zerocopy::project_clients::ProjectDerive,
            ẕfield,
            (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Uninit),
            { ::zerocopy::STRUCT_VARIANT_ID },
            { ::zerocopy::ident_id!(field) },
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
            ::zerocopy::project_clients::ProjectDerive,
            ẕfield,
            (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Initialized),
            { ::zerocopy::STRUCT_VARIANT_ID },
            { ::zerocopy::ident_id!(field) },
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
            ::zerocopy::project_clients::ProjectDerive,
            ẕfield,
            (___ZcAliasing, ___ZcAlignment, ::zerocopy::invariant::Safe),
            { ::zerocopy::STRUCT_VARIANT_ID },
            { ::zerocopy::ident_id!(field) },
        > for Foo {
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
