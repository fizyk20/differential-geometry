use coordinates::{CoordinateSystem, Point};
use std::ops::{Index, IndexMut};
use typenum::uint::Unsigned;
use typenum::Pow;
use generic_array::{GenericArray, ArrayLength};
use super::{CovariantIndex, ContravariantIndex, Variance, IndexType};

/// This is a struct that represents a generic tensor
pub struct Tensor<T: CoordinateSystem, U: Variance>
    where T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
    p: Point<T>,
    x: GenericArray<f64, <T::Dimension as Pow<U::Rank>>::Output>
}

impl<T, U> Clone for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
    fn clone(&self) -> Tensor<T, U> {
        Tensor {
            p: self.p.clone(),
            x: self.x.clone()
        }
    }
}

impl<T, U> Copy for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as ArrayLength<f64>>::ArrayType: Copy,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>,
          <<T::Dimension as Pow<U::Rank>>::Output as ArrayLength<f64>>::ArrayType: Copy 
{}

impl<T, U> Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
    /// Returns the point at which the tensor is defined.
    pub fn get_point(&self) -> &Point<T> {
        &self.p
    }

    /// Returns a reference to a coordinate of the tensor, at indices specified by the slice.
    /// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
    pub fn get_coord(&self, i: &[usize]) -> &f64 {
        assert_eq!(i.len(), U::rank());
        let dim = T::dimension();
        let index = i.into_iter().fold(0, |res, idx| {
            assert!(*idx < dim);
            res * dim + idx
        });
        &self.x[index]
    }

    /// Returns a mutable reference to a coordinate of the tensor, at indices specified by the slice.
    /// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
    pub fn get_coord_mut(&mut self, i: &[usize]) -> &mut f64 {
        assert_eq!(i.len(), U::rank());
        let dim = T::dimension();
        let index = i.into_iter().fold(0, |res, idx| {
            assert!(*idx < dim);
            res * dim + idx
        });
        &mut self.x[index]
    }

    /// Returns the variance of the tensor, that is, the list of the index types.
    /// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
    pub fn get_variance() -> Vec<IndexType> {
        U::variance()
    }

    /// Returns the rank of the tensor
    pub fn get_rank() -> usize {
        U::rank()
    }

    /// Returns the number of coordinates of the tensor
    pub fn get_num_coords() -> usize {
        <T::Dimension as Pow<U::Rank>>::Output::to_usize()
    }

    pub fn new(point: Point<T>) -> Tensor<T, U> {
        Tensor {
            p: point,
            x: GenericArray::new()
        }
    }
}

impl<'a, T, U> Index<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
    type Output = f64;

    fn index(&self, idx: &'a [usize]) -> &f64 {
        self.get_coord(idx)
    }
}

impl<'a, T, U> IndexMut<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
    fn index_mut(&mut self, idx: &'a [usize]) -> &mut f64 {
        self.get_coord_mut(idx)
    }
}

pub type Vector<T> = Tensor<T, ContravariantIndex>;
pub type Covector<T> = Tensor<T, CovariantIndex>;
pub type Matrix<T> = Tensor<T, (ContravariantIndex, CovariantIndex)>;
