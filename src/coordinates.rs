use std::ops::{Index, IndexMut};
use super::tensors::{IndexType, Tensor, GenericTensor};
use std::marker::PhantomData;

/// CoordinateSystem trait marks a struct (usually a unit struct) as representing a coordinate system.
pub trait CoordinateSystem : Sized {	
	/// Function returning the dimension (number of coordinates) for dynamic checks.
	/// This would be better solved with the dimension as a type parameter, but it's not supported as of Rust 1.3.
	fn dimension() -> usize;

	/// Function returning a small value for purposes of numerical differentiation
	fn small() -> f64;
}

/// Struct representing a point on the manifold. The information about the coordinate system is saved in the type parameter,
/// so that only operations on objects in one coordinate system will be allowed.
pub struct Point<T: CoordinateSystem> {
	/// The coordinates of the point.
	x: Vec<f64>,
	phantom: PhantomData<T>
}

impl<T> Point<T> where T: CoordinateSystem {
	
	/// Creates a new point with coordinates described by the slice.
	pub fn new(coords: &[f64]) -> Point<T> {
		assert_eq!(coords.len(), T::dimension());
		Point { x: Vec::from(coords), phantom: PhantomData }
	}
	
}

impl<T> Clone for Point<T> where T: CoordinateSystem {
	
	fn clone(&self) -> Point<T> {
		Point::new(&self.x)
	}
	
}
	
impl<T> Index<usize> for Point<T> where T: CoordinateSystem {
	type Output = f64;
	
	fn index(&self, idx: usize) -> &f64 {
		&self.x[idx]
	}
}

impl<T> IndexMut<usize> for Point<T> where T: CoordinateSystem {
	fn index_mut(&mut self, idx: usize) -> &mut f64 {
		&mut self.x[idx]
	}
}

/// Trait used for conversions between different coordinate systems. Implementing ConversionTo<T> for a CoordinateSystem
/// will allow objects in that system to be converted to the system T (note that T also has to be a CoordinateSystem).
pub trait ConversionTo<T: CoordinateSystem + 'static> : CoordinateSystem
{	
	/// Function converting the coordinates of a point.
	fn convert_point(p: &Point<Self>) -> Point<T>;
	
	/// Function calculating a Jacobian at a point - that is, the matrix of derivatives of the coordinate conversions.
	fn jacobian(p: &Point<Self>) -> Box<Tensor<T>> {
		let d = Self::dimension();
		let mut result = Box::new(GenericTensor::new(vec![IndexType::Contravariant, IndexType::Covariant], Self::convert_point(p)));
		let h = Self::small();

		for j in 0..d {
			let mut x = p.clone();
			x[j] = x[j] - h;
			let y1 = Self::convert_point(&x);

			x[j] = x[j] + h + h;
			let y2 = Self::convert_point(&x);

			for i in 0..d {
				// calculate dyi/dxj
				let index = [i, j];
				result[&index[..]] = (y2[i] - y1[i])/(2.0*h);
			}
		}

		result
	}
	
	/// The inverse matrix of the Jacobian at a point.
	fn inv_jacobian(p: &Point<Self>) -> Tensor<Self>;
}