//! Module containing definitions of tensors and operations on them.
mod tensor;
mod variance;

pub use self::variance::{IndexType, ContravariantIndex, CovariantIndex, TensorIndex, OtherIndex,
                         Variance, Concat, Contract, Joined, Contracted};
pub use self::tensor::{Tensor, Scalar, Vector, Covector, Matrix, TwoForm, InvTwoForm, InnerProduct};
