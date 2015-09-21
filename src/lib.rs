//! **diffgeom** is a crate aiming to leverage the Rust type system to provide
//! a convenient and clean API for tensor calculus on arbitrary manifolds.

pub mod coordinates;
pub mod tensors;

#[cfg(test)]
mod tests;