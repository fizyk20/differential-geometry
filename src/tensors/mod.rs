//! This is a module containing definitions of different tensors
mod tensor;

pub use self::tensor::{IndexType, Tensor, Up, Down, TensorIndex, Variance};
pub use self::tensor::{Vector, Covector, Matrix};