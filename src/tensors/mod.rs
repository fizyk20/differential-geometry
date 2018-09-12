//! Module containing definitions of tensors and operations on them.
mod tensor;
mod variance;

pub use self::tensor::{
    Covector, InnerProduct, InvTwoForm, Matrix, Scalar, Tensor, TwoForm, Vector,
};
pub use self::variance::{
    Concat, Contract, Contracted, ContravariantIndex, CovariantIndex, IndexType, Joined,
    OtherIndex, TensorIndex, Variance,
};
