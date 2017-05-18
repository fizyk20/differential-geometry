//! Module defining variances (types of tensors)
use std::ops::{Add, Sub};
use typenum::uint::{Unsigned, UInt};
use typenum::bit::Bit;
use typenum::consts::{U0, U1, B1};
use typenum::{Cmp, Same, Greater};
use typenum::{Add1, Sub1};

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
    type Rank: Unsigned + Add<B1>;
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
///
/// Used for identifying indices that can be contracted
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
          Add1<U::Rank>: Unsigned + Add<B1>,
          T: TensorIndex
{
    type Rank = Add1<U::Rank>;

    fn variance() -> Vec<IndexType> {
        let mut result = vec![T::index_type()];
        result.append(&mut U::variance());
        result
    }
}

/// Operator trait used for concatenating two variances.
///
/// Used in tensor outer product.
pub trait Concat<T: Variance>: Variance {
    type Output: Variance;
}

/// Helper type for variance concatenation.
pub type Joined<T, U> = <T as Concat<U>>::Output;

impl<T, U> Concat<U> for T
    where T: TensorIndex,
          U: TensorIndex,
          Add1<<U as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = (T, U);
}

impl<T> Concat<T> for ()
    where T: TensorIndex
{
    type Output = T;
}

impl<T, U, V> Concat<V> for (T, U)
    where T: TensorIndex,
          V: TensorIndex,
          U: Variance + Concat<V>,
          <U as Concat<V>>::Output: Variance,
          Add1<<U as Variance>::Rank>: Unsigned + Add<B1>,
          Add1<<Joined<U, V> as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = (T, <U as Concat<V>>::Output);
}

impl<T, U, V> Concat<(U, V)> for T
    where T: TensorIndex,
          U: TensorIndex,
          V: Variance,
          Add1<<V as Variance>::Rank>: Unsigned + Add<B1>,
          Add1<Add1<<V as Variance>::Rank>>: Unsigned + Add<B1>
{
    type Output = (T, (U, V));
}

impl<T, U, V, W> Concat<(V, W)> for (T, U)
    where T: TensorIndex,
          U: Variance + Concat<(V, W)>,
          V: TensorIndex,
          W: Variance,
          Add1<<U as Variance>::Rank>: Unsigned + Add<B1>,
          Add1<<W as Variance>::Rank>: Unsigned + Add<B1>,
          Add1<<Joined<U, (V, W)> as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = (T, Joined<U, (V, W)>);
}

/// Indexing operator trait: Output is equal to the index type at the given position
///
/// Warning: Indices are numbered starting from 0!
pub trait Index<T: Unsigned>: Variance {
    type Output: TensorIndex;
}

/// Helper type for variance indexing.
pub type At<T, U> = <T as Index<U>>::Output;

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
          UInt<U, B>: Sub<B1>,
          Sub1<UInt<U, B>>: Unsigned,
          T: Variance + Index<Sub1<UInt<U, B>>>,
          Add1<<T as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = At<T, Sub1<UInt<U, B>>>;
}

impl<T, V> Index<U0> for (V, T)
    where V: TensorIndex,
          T: Variance,
          Add1<<T as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = V;
}

/// An operator trait, removing the indicated index from a variance
pub trait RemoveIndex<T: Unsigned>: Variance {
    type Output: Variance;
}

/// Helper type for index removal
pub type Removed<T, U> = <T as RemoveIndex<U>>::Output;

impl RemoveIndex<U0> for CovariantIndex {
    type Output = ();
}

impl RemoveIndex<U0> for ContravariantIndex {
    type Output = ();
}

impl<U, V> RemoveIndex<U0> for (U, V)
    where U: TensorIndex,
          V: Variance,
          Add1<<V as Variance>::Rank>: Unsigned + Add<B1>
{
    type Output = V;
}

impl<T, B, U, V> RemoveIndex<UInt<T, B>> for (U, V)
    where T: Unsigned,
          B: Bit,
          U: TensorIndex,
          UInt<T, B>: Sub<B1>,
          Sub1<UInt<T, B>>: Unsigned,
          V: Variance + RemoveIndex<Sub1<UInt<T, B>>>,
          (U, V): Variance,
          (U, Removed<V, Sub1<UInt<T, B>>>): Variance
{
    type Output = (U, Removed<V, Sub1<UInt<T, B>>>);
}

/// An operator trait representing tensor contraction
///
/// Used in tensor inner product
pub trait Contract<Ul: Unsigned, Uh: Unsigned>: Variance {
    type Output: Variance;
}

/// Helper type for contraction
pub type Contracted<V, Ul, Uh> = <V as Contract<Ul, Uh>>::Output;

impl<Ul, Uh, V> Contract<Ul, Uh> for V
    where Ul: Unsigned,
          Uh: Unsigned + Sub<B1> + Cmp<Ul>,
          Sub1<Uh>: Unsigned,
          <Uh as Cmp<Ul>>::Output: Same<Greater>,
          V: Index<Ul> + Index<Uh> + RemoveIndex<Ul>,
          At<V, Ul>: OtherIndex,
          At<V, Uh>: Same<<At<V, Ul> as OtherIndex>::Output>,
          Removed<V, Ul>: RemoveIndex<Sub1<Uh>>,
          Removed<Removed<V, Ul>, Sub1<Uh>>: Variance
{
    type Output = Removed<Removed<V, Ul>, Sub1<Uh>>;
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
        assert_eq!(<Joined<CovariantIndex, ContravariantIndex> as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Contravariant]);

        assert_eq!(<Joined<(CovariantIndex, CovariantIndex), ContravariantIndex> as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Covariant, IndexType::Contravariant]);

        assert_eq!(<Joined<CovariantIndex, (CovariantIndex, ContravariantIndex)> as Variance>::variance(),
            vec![IndexType::Covariant, IndexType::Covariant, IndexType::Contravariant]);

        assert_eq!(<Joined<(ContravariantIndex, CovariantIndex),
                          (CovariantIndex, ContravariantIndex)> as Variance>::variance(),
                   vec![IndexType::Contravariant,
                        IndexType::Covariant,
                        IndexType::Covariant,
                        IndexType::Contravariant]);
    }

    #[test]
    fn test_index() {
        assert_eq!(<At<CovariantIndex, U0> as TensorIndex>::index_type(),
                   IndexType::Covariant);

        assert_eq!(<At<(CovariantIndex, ContravariantIndex), U0> as TensorIndex>::index_type(),
            IndexType::Covariant);

        assert_eq!(<At<(CovariantIndex, ContravariantIndex), U1> as TensorIndex>::index_type(),
            IndexType::Contravariant);

        assert_eq!(<At<(ContravariantIndex, (CovariantIndex, CovariantIndex)), U0> as TensorIndex>::index_type(),
            IndexType::Contravariant);

        assert_eq!(<At<(ContravariantIndex, (CovariantIndex, CovariantIndex)), U2> as TensorIndex>::index_type(),
            IndexType::Covariant);
    }

    #[test]
    fn test_remove() {
        assert_eq!(<Removed<CovariantIndex, U0> as Variance>::variance(),
                   vec![]);

        assert_eq!(<Removed<(CovariantIndex, ContravariantIndex), U0> as Variance>::variance(),
            vec![IndexType::Contravariant]);

        assert_eq!(<Removed<(CovariantIndex, ContravariantIndex), U1> as Variance>::variance(),
            vec![IndexType::Covariant]);

        assert_eq!(<Removed<(ContravariantIndex, (CovariantIndex, CovariantIndex)), U1> as Variance>::variance(),
            vec![IndexType::Contravariant, IndexType::Covariant]);
    }

    #[test]
    fn test_contract() {
        assert_eq!(<Contracted<(CovariantIndex, ContravariantIndex), U0, U1> as Variance>::variance(),
            vec![]);

        assert_eq!(<Contracted<(ContravariantIndex, CovariantIndex), U0, U1> as Variance>::variance(),
            vec![]);

        assert_eq!(<Contracted<(ContravariantIndex, (CovariantIndex, CovariantIndex)), U0, U1> as Variance>::variance(),
            vec![IndexType::Covariant]);

        assert_eq!(<Contracted<(ContravariantIndex, (CovariantIndex, CovariantIndex)), U0, U2> as Variance>::variance(),
            vec![IndexType::Covariant]);
    }
}
