use coordinates::{CoordinateSystem, Point};
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;

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

pub struct Up;
impl TensorIndex for Up {
	fn index_type() -> IndexType { IndexType::Contravariant }
}

pub struct Down;
impl TensorIndex for Down {
	fn index_type() -> IndexType { IndexType::Covariant }
}

pub trait Variance {
	fn rank() -> usize;
	fn variance() -> Vec<IndexType>;
}

impl Variance for Up {
	fn rank() -> usize { 1 }
	fn variance() -> Vec<IndexType> { vec![IndexType::Contravariant] }
}

impl Variance for Down {
	fn rank() -> usize { 1 }
	fn variance() -> Vec<IndexType> { vec![IndexType::Covariant] }
}

impl<T, U> Variance for (T, U) where T: Variance, U: TensorIndex {
	fn rank() -> usize {
		1 + T::rank()
	}

	fn variance() -> Vec<IndexType> {
		let mut result = T::variance();
		result.push(U::index_type());
		result
	}
}

/// This is a struct that represents a generic tensor
pub struct Tensor<T: CoordinateSystem, U: Variance> {
	p: Point<T>,
	x: Vec<f64>,
	phantom: PhantomData<U>
}

impl<T, U> Tensor<T, U> where T: CoordinateSystem, U: Variance {
	
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
	
	pub fn new(point: Point<T>) -> Tensor<T, U> {
		let len = U::rank();
		let num_coords = T::dimension().pow(len as u32);
		let mut coords: Vec<f64> = Vec::with_capacity(num_coords);
		for _ in 0..num_coords {
			coords.push(0.0);
		}
		Tensor {
			p: point,
			x: coords,
			phantom: PhantomData
		}
	}
}
	
impl<'a, T, U> Index<&'a [usize]> for Tensor<T, U> where T: CoordinateSystem, U: Variance {
	type Output = f64;
	
	fn index(&self, idx: &'a [usize]) -> &f64 {
		self.get_coord(idx)
	}
}

impl<'a, T, U> IndexMut<&'a [usize]> for Tensor<T, U> where T: CoordinateSystem, U: Variance {
	fn index_mut(&mut self, idx: &'a [usize]) -> &mut f64 {
		self.get_coord_mut(idx)
	}
}

pub type Vector<T> = Tensor<T, Up>;
pub type Covector<T> = Tensor<T, Down>;
pub type Matrix<T> = Tensor<T, (Up, Down)>;