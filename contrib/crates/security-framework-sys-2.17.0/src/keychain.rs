#[cfg(target_os = "macos")]
use core_foundation_sys::base::CFTypeRef;
#[cfg(target_os = "macos")]
use core_foundation_sys::base::{Boolean, CFTypeID, OSStatus};
#[cfg(target_os = "macos")]
use std::os::raw::{c_char, c_uint, c_void};

#[cfg(target_os = "macos")]
use crate::base::SecKeychainItemRef;
#[cfg(target_os = "macos")]
use crate::base::{SecAccessRef, SecKeychainRef};

#[cfg(target_os = "macos")]
pub const SEC_KEYCHAIN_SETTINGS_VERS1: c_uint = 1;

#[repr(C)]
#[cfg(target_os = "macos")]
pub struct SecKeychainSettings {
    pub version: c_uint,
    pub lockOnSleep: Boolean,
    pub useLockInterval: Boolean,
    pub lockInterval: c_uint,
}

/// Like Apple's headers, it assumes Little Endian,
/// as there are no supported Big Endian machines any more :(
macro_rules! char_lit {
    ($e:expr) => {
        ($e[3] as u32) + (($e[2] as u32) << 8) + (($e[1] as u32) << 16) + (($e[0] as u32) << 24)
    };
}

macro_rules! char_lit_swapped {
    ($e:expr) => {
        ($e[0] as u32) + (($e[1] as u32) << 8) + (($e[2] as u32) << 16) + (($e[3] as u32) << 24)
    };
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum SecProtocolType {
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    FTP = char_lit!(b"ftp "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    FTPAccount = char_lit!(b"ftpa"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTP = char_lit!(b"http"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    IRC = char_lit!(b"irc "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    NNTP = char_lit!(b"nntp"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    POP3 = char_lit!(b"pop3"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    SMTP = char_lit!(b"smtp"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    SOCKS = char_lit!(b"sox "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    IMAP = char_lit!(b"imap"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    LDAP = char_lit!(b"ldap"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    AppleTalk = char_lit!(b"atlk"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    AFP = char_lit!(b"afp "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    Telnet = char_lit!(b"teln"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    SSH = char_lit!(b"ssh "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    FTPS = char_lit!(b"ftps"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTPS = char_lit!(b"htps"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTPProxy = char_lit!(b"htpx"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTPSProxy = char_lit!(b"htsx"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    FTPProxy = char_lit!(b"ftpx"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    CIFS = char_lit!(b"cifs"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    SMB = char_lit!(b"smb "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    RTSP = char_lit!(b"rtsp"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    RTSPProxy = char_lit!(b"rtsx"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    DAAP = char_lit!(b"daap"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    EPPC = char_lit!(b"eppc"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    IPP = char_lit!(b"ipp "),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    NNTPS = char_lit!(b"ntps"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    LDAPS = char_lit!(b"ldps"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    TelnetS = char_lit!(b"tels"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    IMAPS = char_lit!(b"imps"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    IRCS = char_lit!(b"ircs"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    POP3S = char_lit!(b"pops"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    CVSpserver = char_lit!(b"cvsp"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    SVN = char_lit!(b"svn "),
    Any = 0,
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum SecAuthenticationType {
    // [sic] Apple has got two related enums each with a different endianness!
    NTLM = char_lit_swapped!(b"ntlm"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    MSN = char_lit_swapped!(b"msna"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    DPA = char_lit_swapped!(b"dpaa"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    RPA = char_lit_swapped!(b"rpaa"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTPBasic = char_lit_swapped!(b"http"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTTPDigest = char_lit_swapped!(b"httd"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    HTMLForm = char_lit_swapped!(b"form"),
    #[cfg_attr(not(target_os = "macos"), deprecated(note = "macOS only"))]
    Default = char_lit_swapped!(b"dflt"),
    Any = 0,
}

#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SecPreferencesDomain {
    User = 0,
    System = 1,
    Common = 2,
    Dynamic = 3,
}

extern "C" {
    #[cfg(target_os = "macos")]
    pub fn SecKeychainGetTypeID() -> CFTypeID;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainCopyDefault(keychain: *mut SecKeychainRef) -> OSStatus;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainCopyDomainDefault(
        domain: SecPreferencesDomain,
        keychain: *mut SecKeychainRef,
    ) -> OSStatus;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainCreate(
        pathName: *const c_char,
        passwordLength: c_uint,
        password: *const c_void,
        promptUser: Boolean,
        initialAccess: SecAccessRef,
        keychain: *mut SecKeychainRef,
    ) -> OSStatus;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainOpen(pathName: *const c_char, keychain: *mut SecKeychainRef) -> OSStatus;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainUnlock(
        keychain: SecKeychainRef,
        passwordLength: c_uint,
        password: *const c_void,
        usePassword: Boolean,
    ) -> OSStatus;
    #[cfg(target_os = "macos")]
    pub fn SecKeychainFindGenericPassword(
        keychainOrArray: CFTypeRef,
        serviceNameLength: u32,
        serviceName: *const c_char,
        accountNameLength: u32,
        accountName: *const c_char,
        passwordLength: *mut u32,
        passwordData: *mut *mut c_void,
        itemRef: *mut SecKeychainItemRef,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainFindInternetPassword(
        keychainOrArray: CFTypeRef,
        serverNameLength: u32,
        serverName: *const c_char,
        securityDomainLength: u32,
        securityDomain: *const c_char,
        accountNameLength: u32,
        accountName: *const c_char,
        pathLength: u32,
        path: *const c_char,
        port: u16,
        protocol: SecProtocolType,
        authenticationType: SecAuthenticationType,
        passwordLength: *mut u32,
        passwordData: *mut *mut c_void,
        itemRef: *mut SecKeychainItemRef,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainAddGenericPassword(
        keychain: SecKeychainRef,
        serviceNameLength: u32,
        serviceName: *const c_char,
        accountNameLength: u32,
        accountName: *const c_char,
        passwordLength: u32,
        passwordData: *const c_void,
        itemRef: *mut SecKeychainItemRef,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainAddInternetPassword(
        keychain: SecKeychainRef,
        serverNameLength: u32,
        serverName: *const c_char,
        securityDomainLength: u32,
        securityDomain: *const c_char,
        accountNameLength: u32,
        accountName: *const c_char,
        pathLength: u32,
        path: *const c_char,
        port: u16,
        protocol: SecProtocolType,
        authenticationType: SecAuthenticationType,
        passwordLength: u32,
        passwordData: *const c_void,
        itemRef: *mut SecKeychainItemRef,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainSetSettings(
        keychain: SecKeychainRef,
        newSettings: *const SecKeychainSettings,
    ) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainGetUserInteractionAllowed(state: *mut Boolean) -> OSStatus;

    #[cfg(target_os = "macos")]
    pub fn SecKeychainSetUserInteractionAllowed(state: Boolean) -> OSStatus;
}
