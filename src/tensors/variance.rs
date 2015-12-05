use std::ops::{Add, Sub};
use typenum::uint::{Unsigned, UInt};
use typenum::bit::Bit;
use typenum::consts::{U0, U1};
use typenum::{Cmp, Same, Greater};

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IndexType {
    Covariant,
    Contravariant,
}

/// Trait identifying a type as representing a tensor variance. It is implemented for
/// `CovariantIndex`, `ContravariantIndex` and tuples (Index, Variance).
pub trait Variance {
    type Rank: Unsigned + Add<U1>;
    fn rank() -> usize {
        Self::Rank::to_usize()
    }
    fn variance() -> Vec<IndexType>;
}

impl Variance for () {
    type Rank = U0;

    fn variance() -> Vec<IndexType> {
        vec![]
    }
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

/// Trait representing the other index type
pub trait OtherIndex: TensorIndex {
    type Output: TensorIndex;
}

impl OtherIndex for CovariantIndex {
    type Output = ContravariantIndex;
}

impl OtherIndex for ContravariantIndex {
    type Output = CovariantIndex;
}

// Back to implementing Variance

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
pub trait Concat<T: Variance>: Variance {
    type Output: Variance;
}

impl<T, U> Concat<U> for T
    where T: TensorIndex,
          U: TensorIndex,
          <<U as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = (T, U);
}

impl<T, U, V> Concat<V> for (T, U)
    where T: TensorIndex,
          V: TensorIndex,
          U: Variance + Concat<V>,
          <U as Concat<V>>::Output: Variance,
          <<U as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<<U as Concat<V>>::Output as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = (T, <U as Concat<V>>::Output);
}

impl<T, U, V> Concat<(U, V)> for T
    where T: TensorIndex,
          U: TensorIndex,
          V: Variance,
          <<U as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<V as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<<V as Variance>::Rank as Add<U1>>::Output as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = (T, (U, V));
}

impl<T, U, V, W> Concat<(V, W)> for (T, U)
    where T: TensorIndex,
          U: Variance + Concat<(V, W)>,
          V: TensorIndex,
          W: Variance,
          <<U as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<W as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<<U as Concat<(V, W)>>::Output as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = (T, <U as Concat<(V, W)>>::Output);
}

/// Indexing operator trait: Output is equal to the index type at the given position
///
/// Warning: Indices are numbered starting from 0!
pub trait Index<T: Unsigned>: Variance {
    type Output: TensorIndex;
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
          T: Variance + Index<<UInt<U, B> as Sub<U1>>::Output>,
          <<T as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = <T as Index<<UInt<U, B> as Sub<U1>>::Output>>::Output;
}

impl<T, V> Index<U0> for (V, T)
    where V: TensorIndex,
          T: Variance,
          <<T as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = V;
}

/// An operator trait, removing the indicated index from a variance
pub trait RemoveIndex<T: Unsigned>: Variance {
    type Output: Variance;
}

impl RemoveIndex<U0> for CovariantIndex {
    type Output = ();
}

impl RemoveIndex<U0> for ContravariantIndex {
    type Output = ();
}

impl<U, V> RemoveIndex<U0> for (U, V)
    where U: TensorIndex,
          V: Variance,
          <<V as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = V;
}

impl<T, B, U, V> RemoveIndex<UInt<T, B>> for (U, V)
    where T: Unsigned,
          B: Bit,
          U: TensorIndex,
          UInt<T, B>: Sub<U1>,
          <UInt<T, B> as Sub<U1>>::Output: Unsigned,
          V: Variance + RemoveIndex<<UInt<T, B> as Sub<U1>>::Output>,
          <<V as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>,
          <<<V as RemoveIndex<<UInt<T, B> as Sub<U1>>::Output>>::Output as Variance>::Rank as Add<U1>>::Output: Unsigned + Add<U1>
{
    type Output = (U, <V as RemoveIndex<<UInt<T, B> as Sub<U1>>::Output>>::Output);
}

/// An operator trait representing tensor contraction
pub trait Contract<Ul: Unsigned, Uh: Unsigned>: Variance {
    type Output: Variance;
}

// this is quite possibly the worst impl I have ever written
impl<Ul, Uh, V> Contract<Ul, Uh> for V
    where Ul: Unsigned,
          Uh: Unsigned + Sub<U1> + Cmp<Ul>,
          <Uh as Sub<U1>>::Output: Unsigned,
          <Uh as Cmp<Ul>>::Output: Same<Greater>,
          V: Index<Ul> + Index<Uh> + RemoveIndex<Ul>,
          <V as Index<Ul>>::Output: OtherIndex,
          <V as Index<Uh>>::Output: TensorIndex + Same<<<V as Index<Ul>>::Output as OtherIndex>::Output>,
          <V as RemoveIndex<Ul>>::Output: RemoveIndex<<Uh as Sub<U1>>::Output>,
          <<V as RemoveIndex<Ul>>::Output as RemoveIndex<<Uh as Sub<U1>>::Output>>::Output: Variance
{
    type Output = <<V as RemoveIndex<Ul>>::Output as RemoveIndex<<Uh as Sub<U1>>::Output>>::Output;
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

    #[test]
    fn test_remove() {
        assert_eq!(<CovariantIndex as RemoveIndex<U0>>::Output::variance(),
            vec![]);

        assert_eq!(<(CovariantIndex, ContravariantIndex) as RemoveIndex<U0>>::Output::variance(),
            vec![IndexType::Contravariant]);

        assert_eq!(<(CovariantIndex, ContravariantIndex) as RemoveIndex<U1>>::Output::variance(),
            vec![IndexType::Covariant]);

        assert_eq!(<(ContravariantIndex, (CovariantIndex, CovariantIndex)) as RemoveIndex<U1>>::Output::variance(),
            vec![IndexType::Contravariant, IndexType::Covariant]);
    }

    #[test]
    fn test_contract() {
        assert_eq!(<(CovariantIndex, ContravariantIndex) as Contract<U0, U1>>::Output::variance(),
            vec![]);

        assert_eq!(<(ContravariantIndex, CovariantIndex) as Contract<U0, U1>>::Output::variance(),
            vec![]);

        assert_eq!(<(ContravariantIndex, (CovariantIndex, CovariantIndex)) as Contract<U0, U1>>::Output::variance(),
            vec![IndexType::Covariant]);

        assert_eq!(<(ContravariantIndex, (CovariantIndex, CovariantIndex)) as Contract<U0, U2>>::Output::variance(),
            vec![IndexType::Covariant]);
    }
}