//! This is a module containing definitions of different tensors
mod tensor;

pub use self::tensor::{IndexType, Tensor, ContravariantIndex, CovariantIndex, TensorIndex,
                       Variance};
pub use self::tensor::{Vector, Covector, Matrix};
