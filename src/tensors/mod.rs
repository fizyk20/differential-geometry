//! This is a module containing definitions of different tensors
mod tensor;
mod vector;

pub use self::tensor::{IndexType, Tensor, GenericTensor, Up, Down, TensorIndex, Variance};
pub use self::vector::Vector; 