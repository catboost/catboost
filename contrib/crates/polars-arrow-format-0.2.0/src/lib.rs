#![cfg_attr(docsrs, feature(doc_cfg))]
//! Generated code for Apache Arrow IPC and flight specifictions.

#[cfg(feature = "ipc")]
#[cfg_attr(docsrs, doc(cfg(feature = "ipc")))]
pub mod ipc;

#[cfg(any(feature = "flight-data", feature = "flight-service"))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "flight-data", feature = "flight-service")))
)]
pub mod flight;
