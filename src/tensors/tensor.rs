use coordinates::{CoordinateSystem, Point};
use std::ops::{Index, IndexMut, Add};
use std::marker::PhantomData;
use typenum::uint::Unsigned;
use typenum::consts::U1;
use typenum::Pow;
use generic_array::{GenericArray, ArrayLength};

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy)]
pub enum IndexType {
       Covariant,
       Contravariant
}

pub trait TensorIndex {
	fn index_type() -> IndexType;
}

pub struct ContravariantIndex;
impl TensorIndex for ContravariantIndex {
	fn index_type() -> IndexType { IndexType::Contravariant }
}

pub struct CovariantIndex;
impl TensorIndex for CovariantIndex {
	fn index_type() -> IndexType { IndexType::Covariant }
}

pub trait Variance {
	type Rank: Unsigned + Add<U1>;
	fn rank() -> usize { Self::Rank::to_usize() }
	fn variance() -> Vec<IndexType>;
}

impl Variance for ContravariantIndex {
	type Rank = U1;
	fn variance() -> Vec<IndexType> { vec![IndexType::Contravariant] }
}

impl Variance for CovariantIndex {
	type Rank = U1;
	fn variance() -> Vec<IndexType> { vec![IndexType::Covariant] }
}

impl<T, U> Variance for (T, U) 
	where 
		T: Variance,
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

/// This is a struct that represents a generic tensor
pub struct Tensor<T: CoordinateSystem, U: Variance>
    where T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
	p: Point<T>,
	x: GenericArray<f64, <T::Dimension as Pow<U::Rank>>::Output>,
	phantom: PhantomData<U>
}

impl<T, U> Tensor<T, U>
    where T: CoordinateSystem, U: Variance,
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
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&self.x[index]
	}
	
	/// Returns a mutable reference to a coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	pub fn get_coord_mut(&mut self, i: &[usize]) -> &mut f64 {
		assert_eq!(i.len(), U::rank());
		let dim = T::dimension();
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&mut self.x[index]
	}
	
	/// Returns the variance of the tensor, that is, the list of the index types.
	/// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
	pub fn get_variance() -> Vec<IndexType> { U::variance() }

	/// Returns the rank of the tensor
	pub fn get_rank() -> usize { U::rank() }

    /// Returns the number of coordinates of the tensor
    pub fn get_num_coords() -> usize { <T::Dimension as Pow<U::Rank>>::Output::to_usize() }
	
	pub fn new(point: Point<T>) -> Tensor<T, U> {
		Tensor {
			p: point,
			x: GenericArray::new(),
			phantom: PhantomData
		}
	}
}
	
impl<'a, T, U> Index<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem, U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as Pow<U::Rank>>::Output: ArrayLength<f64>
{
	type Output = f64;
	
	fn index(&self, idx: &'a [usize]) -> &f64 {
		self.get_coord(idx)
	}
}

impl<'a, T, U> IndexMut<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem, U: Variance,
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

#[cfg(test)]
mod test {
	use typenum::consts::U4;
	use coordinates::CoordinateSystem;
	use super::{Vector, Matrix};

	struct Test;
	impl CoordinateSystem for Test {
		type Dimension = U4;
	}

	#[test]
	fn test_ranks() {
		assert_eq!(Vector::<Test>::get_rank(), 1);
		assert_eq!(Matrix::<Test>::get_rank(), 2);
	}

	#[test]
	fn test_num_coords() {
		assert_eq!(Vector::<Test>::get_num_coords(), 4);
		assert_eq!(Matrix::<Test>::get_num_coords(), 16);
	}
}
