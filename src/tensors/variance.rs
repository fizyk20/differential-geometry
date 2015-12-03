use std::ops::Add;
use typenum::uint::Unsigned;
use typenum::consts::U1;

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy)]
pub enum IndexType {
    Covariant,
    Contravariant,
}

pub trait TensorIndex {
    fn index_type() -> IndexType;
}

pub struct ContravariantIndex;
impl TensorIndex for ContravariantIndex {
    fn index_type() -> IndexType {
        IndexType::Contravariant
    }
}

pub struct CovariantIndex;
impl TensorIndex for CovariantIndex {
    fn index_type() -> IndexType {
        IndexType::Covariant
    }
}

pub trait Variance {
    type Rank: Unsigned + Add<U1>;
    fn rank() -> usize {
        Self::Rank::to_usize()
    }
    fn variance() -> Vec<IndexType>;
}

impl Variance for ContravariantIndex {
    type Rank = U1;
    fn variance() -> Vec<IndexType> {
        vec![IndexType::Contravariant]
    }
}

impl Variance for CovariantIndex {
    type Rank = U1;
    fn variance() -> Vec<IndexType> {
        vec![IndexType::Covariant]
    }
}

impl<T, U> Variance for (T, U)
    where T: Variance,
          <T::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          U: TensorIndex
{
    type Rank = <T::Rank as Add<U1>>::Output;

    fn variance() -> Vec<IndexType> {
        let mut result = T::variance();
        result.push(U::index_type());
        result
    }
}