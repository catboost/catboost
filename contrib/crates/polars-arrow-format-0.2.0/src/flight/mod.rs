#[cfg(feature = "flight-data")]
#[cfg_attr(docsrs, doc(cfg(feature = "flight-data")))]
pub mod data;

#[cfg(feature = "flight-service")]
#[cfg_attr(docsrs, doc(cfg(feature = "flight-service")))]
pub mod service;
