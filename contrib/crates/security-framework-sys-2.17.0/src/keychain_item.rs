#[cfg(target_os = "macos")]
use crate::base::{SecKeychainAttributeList, SecKeychainItemRef};
#[cfg(target_os = "macos")]
use core_foundation_sys::base::CFTypeID;
use core_foundation_sys::base::{CFTypeRef, OSStatus};
use core_foundation_sys::dictionary::CFDictionaryRef;
#[cfg(target_os = "macos")]
use std::os::raw::c_void;

extern "C" {

    /// Returns the unique identifier of the opaque type to which a keychain item object belongs.
    #[cfg(target_os = "macos")]
    pub fn SecKeychainItemGetTypeID() -> CFTypeID;

    /// Adds one or more items to a keychain.
    pub fn SecItemAdd(attributes: CFDictionaryRef, result: *mut CFTypeRef) -> OSStatus;

    /// Returns one or more keychain items that match a search query, or copies attributes of specific keychain items.
    pub fn SecItemCopyMatching(query: CFDictionaryRef, result: *mut CFTypeRef) -> OSStatus;

    /// Modifies items that match a search query.
    pub fn SecItemUpdate(query: CFDictionaryRef, attributesToUpdate: CFDictionaryRef) -> OSStatus;

    /// Deletes items that match a search query.
    pub fn SecItemDelete(query: CFDictionaryRef) -> OSStatus;

    /// # Legacy API
    #[cfg(target_os = "macos")]
    pub fn SecKeychainItemModifyAttributesAndData(
        itemRef: SecKeychainItemRef,
        attrList: *const SecKeychainAttributeList,
        length: u32,
        data: *const c_void,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainItemFreeContent(
        attrList: *mut SecKeychainAttributeList,
        data: *mut c_void,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainItemDelete(itemRef: SecKeychainItemRef) -> OSStatus;
}
