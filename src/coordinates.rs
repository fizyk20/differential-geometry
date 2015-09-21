use super::tensors::Tensor;

/// CoordinateSystem trait marks a struct (usually a unit struct) as representing a coordinate system.
pub trait CoordinateSystem {
	/// CoordType represents a type used for the coordinates. This way they aren't limited to a single numeric type,
	/// but for example Mpfr can be used if greater precision is needed.
	type CoordType: Clone;
	
	/// Function returning the dimension (number of coordinates) for dynamic checks.
	/// This would be better solved with the dimension as a type parameter, but it's not supported as of Rust 1.3.
	fn dimension() -> u8;
}

/// Struct representing a point on the manifold. The information about the coordinate system is saved in the type parameter,
/// so that only operations on objects in one coordinate system will be allowed.
pub struct Point<T: CoordinateSystem> {
	/// The coordinates of the point.
	x: Vec<T::CoordType>
}

impl<T> Point<T> where T: CoordinateSystem {
	
	/// Creates a new point with coordinates described by the slice.
	pub fn new(coords: &[T::CoordType]) -> Point<T> {
		assert_eq!(coords.len(), T::dimension() as usize);
		Point { x: Vec::from(coords) }
	}
	
}

/// Trait used for conversions between different coordinate systems. Implementing ConversionTo<T> for a CoordinateSystem
/// will allow objects in that system to be converted to the system T (note that T also has to be a CoordinateSystem).
pub trait ConversionTo<T: CoordinateSystem> : CoordinateSystem {
	
	/// Function converting the coordinates of a point.
	fn convert_point(p: &Point<Self>) -> Point<T>;
	
	/// Function calculating a Jacobian at a point - that is, the matrix of derivatives of the coordinate conversions.
	fn jacobian(p: &Point<Self>) -> Tensor<Self>;
	
	/// The inverse matrix of the Jacobian at a point.
	fn inv_jacobian(p: &Point<Self>) -> Tensor<Self>;
}