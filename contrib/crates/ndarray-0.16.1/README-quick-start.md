# Quickstart tutorial

If you are familiar with Python Numpy, do check out this [For Numpy User Doc](https://docs.rs/ndarray/0.13.0/ndarray/doc/ndarray_for_numpy_users/index.html)
after you go through this tutorial. 

You can use [play.integer32.com](https://play.integer32.com/) to immediately try out the examples.

## The Basics

You can create your first 2x3 floating-point ndarray as such: 
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
                [1.,2.,3.], 
                [4.,5.,6.],
            ]; 
    assert_eq!(a.ndim(), 2);         // get the number of dimensions of array a
    assert_eq!(a.len(), 6);          // get the number of elements in array a
    assert_eq!(a.shape(), [2, 3]);   // get the shape of array a
    assert_eq!(a.is_empty(), false); // check if the array has zero elements

    println!("{:?}", a);
}
```
This code will create a simple array, then print it to stdout as such:
```
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]], shape=[2, 3], strides=[3, 1], layout=C (0x1), const ndim=2
```

## Array Creation

### Element type and dimensionality

Now let's create more arrays. A common operation on matrices is to create a matrix full of 0's of certain dimensions. Let's try to do that with dimensions (3, 2, 4) using the `Array::zeros` function:
```rust
use ndarray::prelude::*;
use ndarray::Array;
fn main() {
    let a = Array::zeros((3, 2, 4).f());
    println!("{:?}", a);
}
```
Unfortunately, this code does not compile.
```
|    let a = Array::zeros((3, 2, 4).f());
|        -   ^^^^^^^^^^^^ cannot infer type for type parameter `A`
```
Indeed, note that the compiler needs to infer the element type and dimensionality from context only. In this 
case the compiler does not have enough information. To fix the code, we can explicitly give the element type through turbofish syntax, and let it infer the dimensionality type:

```rust
use ndarray::prelude::*;
use ndarray::Array;
fn main() {
  let a = Array::<f64, _>::zeros((3, 2, 4).f());
  println!("{:?}", a);
}
```
This code now compiles to what we wanted:
```
[[[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]],

 [[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]],

 [[0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0]]], shape=[3, 2, 4], strides=[1, 3, 6], layout=F (0x2), const ndim=3
```

We could also specify its dimensionality explicitly `Array::<f64, Ix3>::zeros(...)`, with`Ix3` standing for 3D array type. Phew! We achieved type safety. If you tried changing the code above to `Array::<f64, Ix3>::zeros((3, 2, 4, 5).f());`, which is not of dimension 3 anymore, Rust's type system would gracefully prevent you from compiling the code.

### Creating arrays with different initial values and/or different types

The [`from_elem`](http://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.from_elem) method allows initializing an array of given dimension to a specific value of any type:

```rust
use ndarray::{Array, Ix3};
fn main() {
  let a = Array::<bool, Ix3>::from_elem((3, 2, 4), false);
  println!("{:?}", a);
}
```

### Some common array initializing helper functions
`linspace` - Create a 1-D array with 11 elements with values 0., …, 5.
```rust
use ndarray::prelude::*;
use ndarray::{Array, Ix3};
fn main() {
  let a = Array::<f64, _>::linspace(0., 5., 11);
  println!("{:?}", a);
}
```
The output is:
```
[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], shape=[11], strides=[1], layout=C | F (0x3), const ndim=1
```

Common array initializing methods include [`range`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html#method.range), [`logspace`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html#method.logspace), [`eye`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html#method.eye), [`ones`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html#method.ones)...

## Basic operations

Basic operations on arrays are all element-wise; you need to use specific methods for operations such as matrix multiplication (see later section).
```rust
use ndarray::prelude::*;
use ndarray::Array;
use std::f64::INFINITY as inf;

fn main() {
    let a = array![
                [10.,20.,30., 40.,], 
            ];
    let b = Array::range(0., 4., 1.);  // [0., 1., 2., 3, ]

    assert_eq!(&a + &b, array![[10., 21., 32., 43.,]]);  // Allocates a new array. Note the explicit `&`.
    assert_eq!(&a - &b, array![[10., 19., 28., 37.,]]);
    assert_eq!(&a * &b, array![[0., 20., 60., 120.,]]);
    assert_eq!(&a / &b, array![[inf, 20., 15., 13.333333333333334,]]);
}
```


Note that (for any binary operator `@`):
* `&A @ &A` produces a new `Array`
* `B @ A` consumes `B`, updates it with the result, and returns it
* `B @ &A` consumes `B`, updates it with the result, and returns it
* `C @= &A` performs an arithmetic operation in place

Try removing all the `&` sign in front of `a` and `b` in the last example: it will not compile anymore because of those rules.

For more info checkout https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#arithmetic-operations



Some operations have `_axis` appended to the function name: they generally take in a parameter of type `Axis` as one of their inputs,
such as `sum_axis`:

```rust
use ndarray::{aview0, aview1, arr2, Axis};

fn main() {
    let a = arr2(&[[1., 2., 3.],
                   [4., 5., 6.]]);
    assert!(
        a.sum_axis(Axis(0)) == aview1(&[5., 7., 9.]) &&
        a.sum_axis(Axis(1)) == aview1(&[6., 15.]) &&

        a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&21.) &&
        a.sum_axis(Axis(0)).sum_axis(Axis(0)) == aview0(&a.sum())
    );
}
```

### Matrix product

```rust
use ndarray::prelude::*;
use ndarray::Array;

fn main() {
    let a = array![
                [10.,20.,30., 40.,], 
            ];
    let b = Array::range(0., 4., 1.);     // b = [0., 1., 2., 3, ]
    println!("a shape {:?}", &a.shape());
    println!("b shape {:?}", &b.shape());
    
    let b = b.into_shape_with_order((4,1)).unwrap(); // reshape b to shape [4, 1]
    println!("b shape after reshape {:?}", &b.shape());
    
    println!("{}", a.dot(&b));            // [1, 4] x [4, 1] -> [1, 1] 
    println!("{}", a.t().dot(&b.t()));    // [4, 1] x [1, 4] -> [4, 4]
}
```
The output is:
```
a shape [1, 4]
b shape [4]
b shape after reshape [4, 1]
[[200]]
[[0, 10, 20, 30],
 [0, 20, 40, 60],
 [0, 30, 60, 90],
 [0, 40, 80, 120]]
```

## Indexing, Slicing and Iterating
One-dimensional arrays can be indexed, sliced and iterated over, much like `numpy` arrays

```rust
use ndarray::prelude::*;
use ndarray::Array;

fn main() {
    let a = Array::range(0., 10., 1.);

    let mut a = a.mapv(|a: f64| a.powi(3));  // numpy equivlant of `a ** 3`; https://doc.rust-lang.org/nightly/std/primitive.f64.html#method.powi

    println!("{}", a);

    println!("{}", a[[2]]);
    println!("{}", a.slice(s![2]));

    println!("{}", a.slice(s![2..5]));

    a.slice_mut(s![..6;2]).fill(1000.);  // numpy equivlant of `a[:6:2] = 1000`
    println!("{}", a);

    for i in a.iter() {
        print!("{}, ", i.powf(1./3.))
    }
}
```
The output is:
```
[0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
8
8
[8, 27, 64]
[1000, 1, 1000, 27, 1000, 125, 216, 343, 512, 729]
9.999999999999998, 1, 9.999999999999998, 3, 9.999999999999998, 4.999999999999999, 5.999999999999999, 6.999999999999999, 7.999999999999999, 8.999999999999998,
```

For more info about iteration see [Loops, Producers, and Iterators](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html#loops-producers-and-iterators)

Let's try a iterating over a 3D array with elements of type `isize`. This is how you index it: 
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
                    [[  0,  1,  2],         // a 3D array  2 x 2 x 3
                     [ 10, 12, 13]],

                    [[100,101,102],
                     [110,112,113]]
                ];

    let a = a.mapv(|a: isize| a.pow(1));  // numpy equivalent of `a ** 1`; 
                                          // This line does nothing except illustrating mapv with isize type 
    println!("a -> \n{}\n", a);

    println!("`a.slice(s![1, .., ..])` -> \n{}\n", a.slice(s![1, .., ..]));

    println!("`a.slice(s![.., .., 2])` -> \n{}\n", a.slice(s![.., .., 2]));

    println!("`a.slice(s![.., 1, 0..2])` -> \n{}\n", a.slice(s![.., 1, 0..2]));

    println!("`a.iter()` ->");
    for i in a.iter() {
        print!("{}, ", i)  // flat out to every element
    }

    println!("\n\n`a.outer_iter()` ->");
    for i in a.outer_iter() {
        print!("row: {}, \n", i)  // iterate through first dimension
    }
}
```
The output is:
```
a -> 
[[[0, 1, 2],
  [10, 12, 13]],

 [[100, 101, 102],
  [110, 112, 113]]]

`a.slice(s![1, .., ..])` -> 
[[100, 101, 102],
 [110, 112, 113]]

`a.slice(s![.., .., 2])` -> 
[[2, 13],
 [102, 113]]

`a.slice(s![.., 1, 0..2])` -> 
[[10, 12],
 [110, 112]]

`a.iter()` ->
0, 1, 2, 10, 12, 13, 100, 101, 102, 110, 112, 113, 

`a.outer_iter()` ->
row: [[0, 1, 2],
 [10, 12, 13]], 
row: [[100, 101, 102],
 [110, 112, 113]], 
```

## Shape Manipulation

### Changing the shape of an array
The shape of an array can be changed with the `into_shape_with_order` or `to_shape` method.

````rust
use ndarray::prelude::*;
use ndarray::Array;
use std::iter::FromIterator;
// use ndarray_rand::RandomExt;
// use ndarray_rand::rand_distr::Uniform;

fn main() {
    // Or you may use ndarray_rand crate to generate random arrays
    // let a = Array::random((2, 5), Uniform::new(0., 10.));
    
    let a = array![
        [3., 7., 3., 4.],
        [1., 4., 2., 2.],
        [7., 2., 4., 9.]];
        
    println!("a = \n{:?}\n", a);
    
    // use trait FromIterator to flatten a matrix to a vector
    let b = Array::from_iter(a.iter());
    println!("b = \n{:?}\n", b);
    
    let c = b.into_shape_with_order([6, 2]).unwrap(); // consume b and generate c with new shape
    println!("c = \n{:?}", c);
}
````
The output is:
```
a = 
[[3.0, 7.0, 3.0, 4.0],
 [1.0, 4.0, 2.0, 2.0],
 [7.0, 2.0, 4.0, 9.0]], shape=[3, 4], strides=[4, 1], layout=C (0x1), const ndim=2

b = 
[3.0, 7.0, 3.0, 4.0, 1.0, 4.0, 2.0, 2.0, 7.0, 2.0, 4.0, 9.0], shape=[12], strides=[1], layout=C | F (0x3), const ndim=1

c = 
[[3.0, 7.0],
 [3.0, 4.0],
 [1.0, 4.0],
 [2.0, 2.0],
 [7.0, 2.0],
 [4.0, 9.0]], shape=[6, 2], strides=[2, 1], layout=C (0x1), const ndim=2
```

### Stacking/concatenating together different arrays

The `stack!` and `concatenate!` macros are helpful for stacking/concatenating
arrays. The `stack!` macro stacks arrays along a new axis, while the
`concatenate!` macro concatenates arrays along an existing axis:

```rust
use ndarray::prelude::*;
use ndarray::{concatenate, stack, Axis};

fn main() {
    let a = array![
        [3., 7., 8.],
        [5., 2., 4.],
    ];

    let b = array![
        [1., 9., 0.],
        [5., 4., 1.],
    ];

    println!("stack, axis 0:\n{:?}\n", stack![Axis(0), a, b]);
    println!("stack, axis 1:\n{:?}\n", stack![Axis(1), a, b]);
    println!("stack, axis 2:\n{:?}\n", stack![Axis(2), a, b]);
    println!("concatenate, axis 0:\n{:?}\n", concatenate![Axis(0), a, b]);
    println!("concatenate, axis 1:\n{:?}\n", concatenate![Axis(1), a, b]);
}
```
The output is:
```
stack, axis 0:
[[[3.0, 7.0, 8.0],
  [5.0, 2.0, 4.0]],

 [[1.0, 9.0, 0.0],
  [5.0, 4.0, 1.0]]], shape=[2, 2, 3], strides=[6, 3, 1], layout=Cc (0x5), const ndim=3

stack, axis 1:
[[[3.0, 7.0, 8.0],
  [1.0, 9.0, 0.0]],

 [[5.0, 2.0, 4.0],
  [5.0, 4.0, 1.0]]], shape=[2, 2, 3], strides=[3, 6, 1], layout=c (0x4), const ndim=3

stack, axis 2:
[[[3.0, 1.0],
  [7.0, 9.0],
  [8.0, 0.0]],

 [[5.0, 5.0],
  [2.0, 4.0],
  [4.0, 1.0]]], shape=[2, 3, 2], strides=[1, 2, 6], layout=Ff (0xa), const ndim=3

concatenate, axis 0:
[[3.0, 7.0, 8.0],
 [5.0, 2.0, 4.0],
 [1.0, 9.0, 0.0],
 [5.0, 4.0, 1.0]], shape=[4, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2

concatenate, axis 1:
[[3.0, 7.0, 8.0, 1.0, 9.0, 0.0],
 [5.0, 2.0, 4.0, 5.0, 4.0, 1.0]], shape=[2, 6], strides=[1, 2], layout=Ff (0xa), const ndim=2
```

### Splitting one array into several smaller ones

More to see here [ArrayView::split_at](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html#method.split_at)
```rust
use ndarray::prelude::*;
use ndarray::Axis;

fn main() {

    let a = array![
        [6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
        [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]];
    
    let (s1, s2) = a.view().split_at(Axis(0), 1);
    println!("Split a from Axis(0), at index 1:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);
    
    
    let (s1, s2) = a.view().split_at(Axis(1), 4);
    println!("Split a from Axis(1), at index 4:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);
}
```
The output is:
```
Split a from Axis(0), at index 1:
s1  = 
[[6, 7, 6, 9, 0, 5, 4, 0, 6, 8, 5, 2]]
s2  = 
[[8, 5, 5, 7, 1, 8, 6, 7, 1, 8, 1, 0]]

Split a from Axis(1), at index 4:
s1  = 
[[6, 7, 6, 9],
 [8, 5, 5, 7]]
s2  = 
[[0, 5, 4, 0, 6, 8, 5, 2],
 [1, 8, 6, 7, 1, 8, 1, 0]]

```

## Copies and Views
### View, Ref or Shallow Copy

Rust has ownership, so we cannot simply update an element of an array while we have a shared view of it. This brings guarantees & helps having more robust code.
```rust
use ndarray::prelude::*;
use ndarray::{Array, Axis};

fn main() {

    let mut a = Array::range(0., 12., 1.).into_shape_with_order([3 ,4]).unwrap();
    println!("a = \n{}\n", a);
    
    {
        let (s1, s2) = a.view().split_at(Axis(1), 2);
        
        // with s as a view sharing the ref of a, we cannot update a here
        // a.slice_mut(s![1, 1]).fill(1234.);
        
        println!("Split a from Axis(0), at index 1:");
        println!("s1  = \n{}", s1);
        println!("s2  = \n{}\n", s2);
    }
    
    // now we can update a again here, as views of s1, s2 are dropped already
    a.slice_mut(s![1, 1]).fill(1234.);
    
    let (s1, s2) = a.view().split_at(Axis(1), 2);
    println!("Split a from Axis(0), at index 1:");
    println!("s1  = \n{}", s1);
    println!("s2  = \n{}\n", s2);
}
```
The output is:
```
a = 
[[0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 10, 11]]

Split a from Axis(0), at index 1:
s1  = 
[[0, 1],
 [4, 5],
 [8, 9]]
s2  = 
[[2, 3],
 [6, 7],
 [10, 11]]

Split a from Axis(0), at index 1:
s1  = 
[[0, 1],
 [4, 1234],
 [8, 9]]
s2  = 
[[2, 3],
 [6, 7],
 [10, 11]]
```

### Deep Copy
As the usual way in Rust, a `clone()` call will
make a copy of your array:
```rust
use ndarray::prelude::*;
use ndarray::Array;

fn main() {

    let mut a = Array::range(0., 4., 1.).into_shape_with_order([2 ,2]).unwrap();
    let b = a.clone();
    
    println!("a = \n{}\n", a);
    println!("b clone of a = \n{}\n", a);
    
    a.slice_mut(s![1, 1]).fill(1234.);
    
    println!("a updated...");
    println!("a = \n{}\n", a);
    println!("b clone of a = \n{}\n", b);
}
```

The output is:
```
a = 
[[0, 1],
 [2, 3]]

b clone of a = 
[[0, 1],
 [2, 3]]

a updated...
a = 
[[0, 1],
 [2, 1234]]

b clone of a = 
[[0, 1],
 [2, 3]]
```

Notice that using `clone()` (or cloning) an `Array` type also copies the array's elements. It creates an independently owned array of the same type.

Cloning an `ArrayView` does not clone or copy the underlying elements - it only clones the view reference (as it happens in Rust when cloning a `&` reference).

## Broadcasting

Arrays support limited broadcasting, where arithmetic operations with array operands of different sizes can be carried out by repeating the elements of the smaller dimension array. 

```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
        [1., 1.], 
        [1., 2.], 
        [0., 3.], 
        [0., 4.]];

    let b = array![[0., 1.]];

    let c = array![
        [1., 2.], 
        [1., 3.], 
        [0., 4.], 
        [0., 5.]];
    
    // We can add because the shapes are compatible even if not equal.
    // The `b` array is shape 1 × 2 but acts like a 4 × 2 array.
    assert!(c == a + b);
}
```

See [.broadcast()](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.broadcast) for a more detailed description.

And here is a short example of it:
```rust
use ndarray::prelude::*;

fn main() {
    let a = array![
        [1., 2.],
        [3., 4.],
    ];
    
    let b =  a.broadcast((3, 2, 2)).unwrap();
    println!("shape of a is {:?}", a.shape());
    println!("a is broadcased to 3x2x2 = \n{}", b);
}
```
The output is:
```
shape of a is [2, 2]
a is broadcased to 3x2x2 = 
[[[1, 2],
  [3, 4]],

 [[1, 2],
  [3, 4]],

 [[1, 2],
  [3, 4]]]
```

## Want to learn more?
Please checkout these docs for more information
* [`ArrayBase` doc page](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)
* [`ndarray` for `numpy` user doc page](https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html)
