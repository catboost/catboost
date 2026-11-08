/// Indexing macro for Dim<[usize; N]> this
/// gets the index at `$i` in the underlying array
macro_rules! get {
    ($dim:expr, $i:expr) => {
        (*$dim.ix())[$i]
    };
}
macro_rules! getm {
    ($dim:expr, $i:expr) => {
        (*$dim.ixm())[$i]
    };
}
