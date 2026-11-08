//! This is a helper crate for the `recursive` crate, containing the procedural macro.

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, ItemFn, ReturnType, Type};

#[proc_macro_attribute]
pub fn recursive(args: TokenStream, item: TokenStream) -> TokenStream {
    let arg_parser = syn::meta::parser(|meta| {
        Err(meta.error("#[recursive] does not have configuration options"))
    });
    parse_macro_input!(args with arg_parser);

    let mut item_fn: ItemFn =
        syn::parse(item.clone()).expect("#[recursive] can only be applied to functions");
    assert!(
        item_fn.sig.asyncness.is_none(),
        "#[recursive] does not support async functions"
    );

    let block = item_fn.block;
    let ret = match &item_fn.sig.output {
        // impl trait is not supported in closure return type, override with
        // default, which is inferring.
        ReturnType::Type(_, typ) if matches!(**typ, Type::ImplTrait(_)) => ReturnType::Default,
        _ => item_fn.sig.output.clone(),
    };
    let wrapped_block = quote! {
        {
            ::recursive::__impl::stacker::maybe_grow(
                ::recursive::get_minimum_stack_size(),
                ::recursive::get_stack_allocation_size(),
                move || #ret { #block }
            )
        }
    };
    item_fn.block = Box::new(syn::parse(wrapped_block.into()).unwrap());
    item_fn.into_token_stream().into()
}
