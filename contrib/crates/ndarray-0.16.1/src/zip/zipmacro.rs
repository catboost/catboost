/// Array zip macro: lock step function application across several arrays and
/// producers.
///
/// This is a shorthand for [`Zip`](crate::Zip).
///
/// This example:
///
/// ```rust,ignore
/// azip!((a in &mut a, &b in &b, &c in &c) *a = b + c);
/// ```
///
/// Is equivalent to:
///
/// ```rust,ignore
/// Zip::from(&mut a).and(&b).and(&c).for_each(|a, &b, &c| {
///     *a = b + c
/// });
/// ```
///
/// The syntax is either
///
/// `azip!((` *pat* `in` *expr* `,` *[* *pat* `in` *expr* `,` ... *]* `)` *body_expr* `)`
///
/// or, to use `Zip::indexed` instead of `Zip::from`,
///
/// `azip!((index` *pat* `,` *pat* `in` *expr* `,` *[* *pat* `in` *expr* `,` ... *]* `)` *body_expr* `)`
///
/// The *expr* are expressions whose types must implement `IntoNdProducer`, the
/// *pat* are the patterns of the parameters to the closure called by
/// `Zip::for_each`, and *body_expr* is the body of the closure called by
/// `Zip::for_each`. You can think of each *pat* `in` *expr* as being analogous to
/// the `pat in expr` of a normal loop `for pat in expr { statements }`: a
/// pattern, followed by `in`, followed by an expression that implements
/// `IntoNdProducer` (analogous to `IntoIterator` for a `for` loop).
///
/// **Panics** if any of the arrays are not of the same shape.
///
/// ## Examples
///
/// ```rust
/// use ndarray::{azip, Array1, Array2, Axis};
///
/// type M = Array2<f32>;
///
/// // Setup example arrays
/// let mut a = M::zeros((16, 16));
/// let mut b = M::zeros(a.dim());
/// let mut c = M::zeros(a.dim());
///
/// // assign values
/// b.fill(1.);
/// for ((i, j), elt) in c.indexed_iter_mut() {
///     *elt = (i + 10 * j) as f32;
/// }
///
/// // Example 1: Compute a simple ternary operation:
/// // elementwise addition of b and c, stored in a
/// azip!((a in &mut a, &b in &b, &c in &c) *a = b + c);
///
/// assert_eq!(a, &b + &c);
///
/// // Example 2: azip!() with index
/// azip!((index (i, j), &b in &b, &c in &c) {
///     a[[i, j]] = b - c;
/// });
///
/// assert_eq!(a, &b - &c);
///
///
/// // Example 3: azip!() on references
/// // See the definition of the function below
/// borrow_multiply(&mut a, &b, &c);
///
/// assert_eq!(a, &b * &c);
///
///
/// // Since this function borrows its inputs, the `IntoNdProducer`
/// // expressions don't need to explicitly include `&mut` or `&`.
/// fn borrow_multiply(a: &mut M, b: &M, c: &M) {
///     azip!((a in a, &b in b, &c in c) *a = b * c);
/// }
///
///
/// // Example 4: using azip!() without dereference in pattern.
/// //
/// // Create a new array `totals` with one entry per row of `a`.
/// // Use azip to traverse the rows of `a` and assign to the corresponding
/// // entry in `totals` with the sum across each row.
/// //
/// // The row is an array view; it doesn't need to be dereferenced.
/// let mut totals = Array1::zeros(a.nrows());
/// azip!((totals in &mut totals, row in a.rows()) *totals = row.sum());
///
/// // Check the result against the built in `.sum_axis()` along axis 1.
/// assert_eq!(totals, a.sum_axis(Axis(1)));
/// ```
#[macro_export]
macro_rules! azip {
    // Indexed with a single producer
    // we allow an optional trailing comma after the producers in each rule.
    (@build $apply:ident (index $index:pat, $first_pat:pat in $first_prod:expr $(,)?) $body:expr) => {
        $crate::Zip::indexed($first_prod).$apply(|$index, $first_pat| $body)
    };
    // Indexed with more than one producer
    (@build $apply:ident (index $index:pat, $first_pat:pat in $first_prod:expr, $($pat:pat in $prod:expr),* $(,)?) $body:expr) => {
        $crate::Zip::indexed($first_prod)
            $(.and($prod))*
            .$apply(|$index, $first_pat, $($pat),*| $body)
    };
    // Unindexed with a single producer
    (@build $apply:ident ($first_pat:pat in $first_prod:expr $(,)?) $body:expr) => {
        $crate::Zip::from($first_prod).$apply(|$first_pat| $body)
    };
    // Unindexed with more than one producer
    (@build $apply:ident ($first_pat:pat in $first_prod:expr, $($pat:pat in $prod:expr),* $(,)?) $body:expr) => {
        $crate::Zip::from($first_prod)
            $(.and($prod))*
            .$apply(|$first_pat, $($pat),*| $body)
    };

    // Unindexed with one or more producer, no loop body
    (@build $apply:ident $first_prod:expr $(, $prod:expr)* $(,)?) => {
        $crate::Zip::from($first_prod)
            $(.and($prod))*
    };
    // catch-all rule
    (@build $($t:tt)*) => { compile_error!("Invalid syntax in azip!()") };
    ($($t:tt)*) => {
        $crate::azip!(@build for_each $($t)*)
    };
}
