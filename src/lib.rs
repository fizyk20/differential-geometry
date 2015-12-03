//! **diffgeom** is a crate aiming to leverage the Rust type system to provide
//! a convenient and clean API for tensor calculus on arbitrary manifolds.

extern crate typenum;
extern crate generic_array;

pub mod coordinates;
pub mod tensors;

#[cfg(test)]
mod tests;
