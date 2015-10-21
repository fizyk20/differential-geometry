//! This is a module containing definitions of different tensors
mod tensor;
mod vector;

pub use self::tensor::{IndexType, Tensor, GenericTensor};
pub use self::vector::Vector; 