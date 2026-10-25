use core_foundation_sys::array::CFArrayRef;
#[cfg(target_os = "macos")]
use core_foundation_sys::base::CFTypeRef;
use core_foundation_sys::base::OSStatus;
use core_foundation_sys::data::CFDataRef;
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::string::CFStringRef;
#[cfg(target_os = "macos")]
use std::os::raw::c_uint;

#[cfg(target_os = "macos")]
use crate::base::{SecAccessRef, SecKeychainRef};

#[cfg(target_os = "macos")]
/// <https://developer.apple.com/documentation/security/secexternalformat>
pub type SecExternalFormat = u32;
#[cfg(target_os = "macos")]
pub type SecExternalItemType = u32;
#[cfg(target_os = "macos")]
pub type SecItemImportExportFlags = u32;
#[cfg(target_os = "macos")]
pub type SecKeyImportExportFlags = u32;

#[cfg(target_os = "macos")]
pub const kSecKeyImportOnlyOne: SecKeyImportExportFlags = 1;
#[cfg(target_os = "macos")]
pub const kSecKeySecurePassphrase: SecKeyImportExportFlags = 2;
#[cfg(target_os = "macos")]
pub const kSecKeyNoAccessControl: SecKeyImportExportFlags = 4;

#[cfg(target_os = "macos")]
pub const SEC_KEY_IMPORT_EXPORT_PARAMS_VERSION: c_uint = 0;

#[cfg(target_os = "macos")]
mod sec_external_format {
    use super::SecExternalFormat;
    pub const kSecFormatUnknown: SecExternalFormat = 0;
    /// a.k.a. X509 for public keys
    pub const kSecFormatOpenSSL: SecExternalFormat = 1;
    /// OpenSSH v.1
    pub const kSecFormatSSH: SecExternalFormat = 2;
    pub const kSecFormatBSAFE: SecExternalFormat = 3;
    /// raw unformatted key bits
    pub const kSecFormatRawKey: SecExternalFormat = 4;
    pub const kSecFormatWrappedPKCS8: SecExternalFormat = 5;
    /// traditional openssl
    pub const kSecFormatWrappedOpenSSL: SecExternalFormat = 6;
    /// OpenSSH v.1
    pub const kSecFormatWrappedSSH: SecExternalFormat = 7;
    pub const kSecFormatWrappedLSH: SecExternalFormat = 8;
    /// DER encoded
    pub const kSecFormatX509Cert: SecExternalFormat = 9;
    /// sequence of certs and/or keys, implies PEM
    pub const kSecFormatPEMSequence: SecExternalFormat = 10;
    /// sequence of certs
    pub const kSecFormatPKCS7: SecExternalFormat = 11;
    /// set of certs and private keys
    pub const kSecFormatPKCS12: SecExternalFormat = 12;
    /// sequence of certs, form netscape-cert-sequence
    pub const kSecFormatNetscapeCertSequence: SecExternalFormat = 13;
    /// OpenSSH v.2
    pub const kSecFormatSSHv2: SecExternalFormat = 14;
}

#[cfg(target_os = "macos")]
pub use sec_external_format::*;

#[repr(C)]
#[derive(Copy, Clone)]
#[cfg(target_os = "macos")]
pub struct SecItemImportExportKeyParameters {
    pub version: c_uint,
    pub flags: SecKeyImportExportFlags,
    pub passphrase: CFTypeRef,
    pub alertTitle: CFStringRef,
    pub alertPrompt: CFStringRef,
    pub accessRef: SecAccessRef,
    pub keyUsage: CFArrayRef,
    pub keyAttributes: CFArrayRef,
}

extern "C" {
    #[cfg(target_os = "macos")]
    pub fn SecItemImport(
        importedData: CFDataRef,
        fileNameOrExtension: CFStringRef,
        inputFormat: *mut SecExternalFormat,
        itemType: *mut SecExternalItemType,
        flags: SecItemImportExportFlags,
        keyParams: *const SecItemImportExportKeyParameters,
        importKeychain: SecKeychainRef,
        outItems: *mut CFArrayRef,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecItemExport(
        secItemOrArray: CFTypeRef,
        outputFormat: SecExternalFormat,
        flags: SecItemImportExportFlags,
        keyParams: *const SecItemImportExportKeyParameters,
        exportedData: *mut CFDataRef,
    ) -> OSStatus;

    pub static kSecImportExportPassphrase: CFStringRef;
    #[cfg(target_os = "macos")]
    pub static kSecImportExportKeychain: CFStringRef;
    #[cfg(target_os = "macos")]
    pub static kSecImportExportAccess: CFStringRef;

    pub static kSecImportItemLabel: CFStringRef;
    pub static kSecImportItemKeyID: CFStringRef;
    pub static kSecImportItemTrust: CFStringRef;
    pub static kSecImportItemCertChain: CFStringRef;
    pub static kSecImportItemIdentity: CFStringRef;

    pub fn SecPKCS12Import(
        pkcs12_data: CFDataRef,
        options: CFDictionaryRef,
        items: *mut CFArrayRef,
    ) -> OSStatus;
}
