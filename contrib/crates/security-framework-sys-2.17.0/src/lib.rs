#![allow(bad_style)]

#[cfg_attr(target_vendor = "apple", link(name = "Security", kind = "framework"))]
extern "C" {}

/// macOS only
#[cfg(target_os = "macos")]
pub mod access;
pub mod access_control;
/// macOS only
#[cfg(target_os = "macos")]
pub mod authorization;
pub mod base;
pub mod certificate;
/// macOS only
#[cfg(target_os = "macos")]
pub mod certificate_oids;
pub mod cipher_suite;
/// macOS only
#[cfg(target_os = "macos")]
pub mod cms;
/// macOS only
#[cfg(target_os = "macos")]
pub mod code_signing;
/// macOS only
#[cfg(target_os = "macos")]
pub mod digest_transform;
/// macOS only
#[cfg(target_os = "macos")]
pub mod encrypt_transform;
pub mod identity;
pub mod import_export;
pub mod item;
pub mod key;
/// macOS only
pub mod keychain;
pub mod keychain_item;
pub mod policy;
pub mod random;
pub mod secure_transport;
/// macOS only
#[cfg(target_os = "macos")]
pub mod transform;
pub mod trust;
/// macOS only
#[cfg(target_os = "macos")]
pub mod trust_settings;
