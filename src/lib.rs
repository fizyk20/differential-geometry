/*!
**diffgeom** is a crate aiming to leverage the Rust type system to provide
a type-safe API for tensor calculus on arbitrary manifolds.

What is tensor calculus?
========================

Tensors are, in a way, a generalized idea similar to vectors and matrices. They are
multidimensional arrays of numbers, but not all such arrays are tensors. What makes
them tensors is how they behave in coordinate transformations. The details are a
topic for a whole academic lecture, so I won't go into them. What's important
is that tensors can be used for describing properties of curved spaces and it is
the intended use case of this crate.

Problems
========

Unfortunately, Rust currently doesn't support generics over static values, so
another representation of type-level numbers is required. In this crate one
provided by [typenum](https://github.com/paholg/typenum) is being used. This
makes it necessary to use a lot of trait bounds, which break the compiler in
a few ways, so some operations require the usage of a pretty cumbersome syntax.

Example
=======

Below you can see a code sample presenting some simple operations.

```
# extern crate diffgeom;
# extern crate typenum;
# #[macro_use]
# extern crate generic_array;
use std::ops::Mul;
use generic_array::{GenericArray, ArrayLength};
use diffgeom::coordinates::{CoordinateSystem, Point};
use diffgeom::tensors::{Vector, Covector, Matrix, InnerProduct};
use typenum::consts::{U0, U1};

fn main() {
    // First, a coordinate system must be defined
    struct SomeSystem;
    impl CoordinateSystem for SomeSystem {
        type Dimension = typenum::consts::U2;    // a two-dimensional coordinate system
    }

    // Each tensor should be anchored at a point, so let's create one
    let point = Point::<SomeSystem>::new(arr![f64; 0.0, 0.0]);

    // A vector can be defined like that:
    let vector = Vector::<SomeSystem>::new(point, arr![f64; 1.0, 2.0]);

    // There are also covectors
    let covector = Covector::<SomeSystem>::new(point, arr![f64; 2.0, 0.5]);

    // They can be multiplied, yielding a matrix
    let matrix = <Vector<SomeSystem> as Mul<Covector<SomeSystem>>>::mul(vector, covector);
    // Unfortunately this causes infinite recursion in the compiler:
    // let matrix = vector * covector;

    // They can be contracted
    let scalar = <Vector<SomeSystem> as InnerProduct<Covector<SomeSystem>, U0, U1>>::inner_product(vector, covector);

    assert_eq!(*scalar, *matrix.trace::<U0, U1>());  // scalars returned by tensor functions need to be dereffed to f64
}
```
*/
extern crate typenum;
#[macro_use]
extern crate generic_array;

pub mod coordinates;
pub mod tensors;
pub mod metric;

#[cfg(test)]
mod tests;
