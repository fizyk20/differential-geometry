use coordinates::{CoordinateSystem, Point, Zero};
use std::ops::{Index, IndexMut};

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
#[derive(Clone, Copy)]
pub enum IndexType {
	Covariant,
	Contravariant
}

/// This is a trait that represents the basic properties of a tensor - having coordinates, a defined rank and being defined at a
/// point of the manifold.
/// Tensor-ness being a trait will allow type like vectors and matrices be incompatible on the level of concrete types, but still compatible
/// as trait objects, so that functions will be able to specify if they accept a particular type of tensor, or any type.
pub trait Tensor<T: CoordinateSystem> {
	
	/// Returns the point at which the tensor is defined.
	fn get_point(&self) -> &Point<T>;
	
	/// Returns a reference to a coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	fn get_coord(&self, i: &[usize]) -> &T::CoordType;
	
	/// Returns a mutable reference to a coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	fn get_coord_mut(&mut self, i: &[usize]) -> &mut T::CoordType; 
	
	/// Returns the rank of the tensor, that is, the list of the index types.
	/// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
	fn get_rank(&self) -> Vec<IndexType>;
	
}

/// A generic tensor structure, representing a tensor with an arbitrary rank
pub struct GenericTensor<T: CoordinateSystem> {
	rank: Vec<IndexType>,
	p: Point<T>,
	x: Vec<T::CoordType>
}

impl<T> Tensor<T> for GenericTensor<T> where T: CoordinateSystem {
	
	fn get_point(&self) -> &Point<T> {
		&self.p
	}
	
	fn get_coord(&self, i: &[usize]) -> &T::CoordType {
		assert_eq!(i.len(), self.rank.len());
		let dim = T::dimension();
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&self.x[index]
	}
	
	fn get_coord_mut(&mut self, i: &[usize]) -> &mut T::CoordType {
		assert_eq!(i.len(), self.rank.len());
		let dim = T::dimension();
		let index = i.into_iter().fold(0, |res, idx| { assert!(*idx < dim); res*dim + idx });
		&mut self.x[index]
	}
	
	fn get_rank(&self) -> Vec<IndexType> {
		self.rank.clone()
	}
}
	
impl<'a, T> Index<&'a [usize]> for GenericTensor<T> where T: CoordinateSystem {
	type Output = T::CoordType;
	
	fn index(&self, idx: &'a [usize]) -> &T::CoordType {
		self.get_coord(idx)
	}
}

impl<'a, T> IndexMut<&'a [usize]> for GenericTensor<T> where T: CoordinateSystem {
	fn index_mut(&mut self, idx: &'a [usize]) -> &mut T::CoordType {
		self.get_coord_mut(idx)
	}
}

impl<T> GenericTensor<T> where T: CoordinateSystem {
	
	pub fn new(rank: Vec<IndexType>, point: Point<T>) -> GenericTensor<T> {
		let len = rank.len() as u32;
		let num_coords = T::dimension().pow(len);
		let mut coords: Vec<T::CoordType> = Vec::with_capacity(num_coords);
		for _ in 0..num_coords {
			coords.push(T::CoordType::zero());
		}
		GenericTensor {
			rank: rank,
			p: point,
			x: coords
		}
	}
}