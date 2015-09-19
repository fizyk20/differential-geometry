use super::tensors::Tensor;

pub trait CoordinateSystem {
	type CoordType: Clone;		// this will let people use types they like (f32, f64, Mpfr, whatever)
	fn dimension() -> u8;
}

pub struct Point<T: CoordinateSystem> {
	x: Vec<T::CoordType>
}

impl<T> Point<T> where T: CoordinateSystem {
	
	pub fn new(coords: &[T::CoordType]) -> Point<T> {
		assert_eq!(coords.len(), T::dimension() as usize);
		Point { x: Vec::from(coords) }
	}
	
}

pub trait ConversionTo<T: CoordinateSystem> : CoordinateSystem {
	fn convert_point(p: &Point<Self>) -> Point<T>;
	
	fn jacobian(p: &Point<Self>) -> Tensor<Self>;
	fn inv_jacobian(p: &Point<Self>) -> Tensor<Self>;
}