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

/// This is a trait that represents the basic properties of a tensor - having coordinates, a defined rank and being defined at a
/// point of the manifold.
/// Tensor-ness being a trait will allow type like vectors and matrices be incompatible on the level of concrete types, but still compatible
/// as trait objects, so that functions will be able to specify if they accept a particular type of tensor, or any type.
pub trait Tensor<T: CoordinateSystem, U: Variance> {
	
	/// Returns the point at which the tensor is defined.
	fn get_point(&self) -> &Point<T>;
	
	/// Returns a reference to a coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	fn get_coord(&self, i: &[usize]) -> &f64;
	
	/// Returns a mutable reference to a coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	fn get_coord_mut(&mut self, i: &[usize]) -> &mut f64; 
	
	/// Returns the variance of the tensor, that is, the list of the index types.
	/// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
	fn get_variance() -> Vec<IndexType> { U::variance() }

	/// Returns the rank of the tensor
	fn get_rank() -> usize { U::rank() }
	
}

/// A generic tensor structure, representing a tensor with an arbitrary rank
pub struct GenericTensor<T: CoordinateSystem, U: Variance> {
	p: Point<T>,
	x: Vec<f64>,
	phantom: PhantomData<U>
}

impl<T, U> Tensor<T, U> for GenericTensor<T, U> where T: CoordinateSystem, U: Variance {
	
	fn get_point(&self) -> &Point<T> {
		&self.p
	}
	
	fn get_coord(&self, i: &[usize]) -> &f64 {
		assert_eq!(i.len(), U::rank());
		let dim = T::dimension();
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&self.x[index]
	}
	
	fn get_coord_mut(&mut self, i: &[usize]) -> &mut f64 {
		assert_eq!(i.len(), U::rank());
		let dim = T::dimension();
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&mut self.x[index]
	}
}
	
impl<'a, T, U> Index<&'a [usize]> for GenericTensor<T, U> where T: CoordinateSystem, U: Variance {
	type Output = f64;
	
	fn index(&self, idx: &'a [usize]) -> &f64 {
		self.get_coord(idx)
	}
}

impl<'a, T, U> IndexMut<&'a [usize]> for GenericTensor<T, U> where T: CoordinateSystem, U: Variance {
	fn index_mut(&mut self, idx: &'a [usize]) -> &mut f64 {
		self.get_coord_mut(idx)
	}
}

impl<T, U> GenericTensor<T, U> where T: CoordinateSystem, U: Variance {
	
	pub fn new(point: Point<T>) -> GenericTensor<T, U> {
		let len = U::rank();
		let num_coords = T::dimension().pow(len as u32);
		let mut coords: Vec<f64> = Vec::with_capacity(num_coords);
		for _ in 0..num_coords {
			coords.push(0.0);
		}
		GenericTensor {
			p: point,
			x: coords,
			phantom: PhantomData
		}
	}
}