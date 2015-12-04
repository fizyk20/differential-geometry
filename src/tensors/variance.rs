use std::ops::Add;
use typenum::uint::Unsigned;
use typenum::consts::U1;

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IndexType {
    Covariant,
    Contravariant,
}

pub trait Variance {
    type Rank: Unsigned + Add<U1>;
    fn rank() -> usize {
        Self::Rank::to_usize()
    }
    fn variance() -> Vec<IndexType>;
}

pub trait TensorIndex: Variance {
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

pub trait Concat<T> {
    type Output;
}

impl<T, U> Concat<U> for T
    where T: TensorIndex,
          U: TensorIndex
{
    type Output = (T, U);
}

impl<T, U, V> Concat<V> for (T, U)
    where T: Variance,
          U: TensorIndex,
          V: TensorIndex
{
    type Output = ((T, U), V);
}

impl<T, U, V> Concat<(U, V)> for T
    where T: TensorIndex + Concat<U>,
          U: Variance,
          V: TensorIndex,
          <T as Concat<U>>::Output: Concat<V>
{
    type Output = <<T as Concat<U>>::Output as Concat<V>>::Output;
}

impl<T, U, V, W> Concat<(V, W)> for (T, U)
    where T: Variance,
          U: TensorIndex,
          (T, U): Concat<V>,
          V: Variance,
          W: TensorIndex
{
    type Output = (<(T, U) as Concat<V>>::Output, W);
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_variance() {
        assert_eq!(<(CovariantIndex, ContravariantIndex) as Variance>::variance(), vec![IndexType::Covariant, IndexType::Contravariant]);
    }
    
    #[test]
    fn test_variance_concat() {
        assert_eq!(<<CovariantIndex as Concat<ContravariantIndex>>::Output as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Contravariant]);
            
        assert_eq!(<<(CovariantIndex, CovariantIndex) as Concat<ContravariantIndex>>::Output as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Covariant, IndexType::Contravariant]);
            
        assert_eq!(<<CovariantIndex as Concat<(CovariantIndex, ContravariantIndex)>>::Output as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Covariant, IndexType::Contravariant]);
            
        assert_eq!(<<(ContravariantIndex, CovariantIndex) as Concat<(CovariantIndex, ContravariantIndex)>>::Output as Variance>::variance(),
            vec![IndexType::Contravariant, IndexType::Covariant, IndexType::Covariant, IndexType::Contravariant]);
    }
}