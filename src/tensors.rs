use super::coordinates::{CoordinateSystem, Point};

/// This enum serves to represent the type of a tensor. A tensor can have any number of indices, and each one can be either
/// covariant (a lower index), or contravariant (an upper index). For example, a vector is a tensor with only one contravariant index.
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
	
	/// Returns the coordinate of the tensor, at indices specified by the slice.
	/// The length of the slice (the number of indices) has to be compatible with the rank of the tensor. 
	fn get_coord(&self, i: &[u8]) -> T::CoordType;
	
	/// Returns the rank of the tensor, that is, the list of the index types.
	/// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
	fn get_rank(&self) -> Vec<IndexType>;
	
}