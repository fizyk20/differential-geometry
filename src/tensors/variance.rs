use std::ops::{Add, Sub};
use typenum::uint::{Unsigned, UInt};
use typenum::bit::Bit;
use typenum::consts::{U0, U1};

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IndexType {
    Covariant,
    Contravariant,
}

/// Trait identifying a type as representing a tensor variance. It is implemented for
/// `CovariantIndex`, `ContravariantIndex` and tuples (Variance, Index).
pub trait Variance {
    type Rank: Unsigned + Add<U1>;
    fn rank() -> usize {
        Self::Rank::to_usize()
    }
    fn variance() -> Vec<IndexType>;
}

/// Trait identifying a type as representing a tensor index. It is implemented
/// for `CovariantIndex` and `ContravariantIndex`.
pub trait TensorIndex: Variance {
    fn index_type() -> IndexType;
}

/// Type representing a contravariant (upper) tensor index.
pub struct ContravariantIndex;
impl TensorIndex for ContravariantIndex {
    fn index_type() -> IndexType {
        IndexType::Contravariant
    }
}

/// Type representing a covariant (lower) tensor index.
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
    where U: Variance,
          <U::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          T: TensorIndex
{
    type Rank = <U::Rank as Add<U1>>::Output;

    fn variance() -> Vec<IndexType> {
        let mut result = vec![T::index_type()];
        result.append(&mut U::variance());
        result
    }
}

/// Operator trait used for concatenating two variances.
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
    where T: TensorIndex,
          V: TensorIndex,
          U: Variance + Concat<V>,
          <U as Concat<V>>::Output: Variance
{
    type Output = (T, <U as Concat<V>>::Output);
}

impl<T, U, V> Concat<(U, V)> for T
    where T: TensorIndex,
          U: TensorIndex,
          V: Variance
{
    type Output = (T, (U, V));
}

impl<T, U, V, W> Concat<(V, W)> for (T, U)
    where T: TensorIndex,
          U: Variance + Concat<(V, W)>,
          V: TensorIndex,
          W: Variance
{
    type Output = (T, <U as Concat<(V, W)>>::Output);
}

/// Indexing operator trait: Output is equal to the index type at the given position
///
/// Warning: Indices are numbered starting from 0!
pub trait Index<T: Unsigned> {
    type Output;
}

impl Index<U0> for CovariantIndex {
    type Output = CovariantIndex;
}

impl Index<U0> for ContravariantIndex {
    type Output = ContravariantIndex;
}

impl<T, V, U, B> Index<UInt<U, B>> for (V, T)
    where V: TensorIndex,
          U: Unsigned,
          B: Bit,
          UInt<U, B>: Sub<U1>,
          <UInt<U, B> as Sub<U1>>::Output: Unsigned,
          T: Variance + Index<<UInt<U, B> as Sub<U1>>::Output>
{
    type Output = <T as Index<<UInt<U, B> as Sub<U1>>::Output>>::Output;
}

impl<T, V> Index<U0> for (V, T)
    where V: TensorIndex,
          T: Variance
{
    type Output = V;
}

#[cfg(test)]
mod test {
    use super::*;
    use typenum::consts::{U0, U1, U2};
    
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

    #[test]
    fn test_index() {
        assert_eq!(<CovariantIndex as Index<U0>>::Output::index_type(),
            IndexType::Covariant);

        assert_eq!(<(CovariantIndex, ContravariantIndex) as Index<U0>>::Output::index_type(),
            IndexType::Covariant);

        assert_eq!(<(CovariantIndex, ContravariantIndex) as Index<U1>>::Output::index_type(),
            IndexType::Contravariant);

        assert_eq!(<(ContravariantIndex, (CovariantIndex, CovariantIndex)) as Index<U0>>::Output::index_type(),
            IndexType::Contravariant);

        assert_eq!(<(ContravariantIndex, (CovariantIndex, CovariantIndex)) as Index<U2>>::Output::index_type(),
            IndexType::Covariant);
    }
}