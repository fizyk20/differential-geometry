use coordinates::{CoordinateSystem, Point};
use super::tensor::{Tensor, IndexType};
use std::ops::{Index, IndexMut};

/// Struct representing a vector (that is, a rank-1 contravariant tensor)
pub struct Vector<T: CoordinateSystem> {
	p: Point<T>,
	x: Vec<f64>
}

impl<T> Tensor<T> for Vector<T> where T: CoordinateSystem {
	
	fn get_point(&self) -> &Point<T> {
		&self.p
	}
	
	fn get_coord(&self, i: &[usize]) -> &f64 {
		assert_eq!(i.len(), 1);
		&self.x[i[0]]
	}
	
	fn get_coord_mut(&mut self, i: &[usize]) -> &mut f64 {
		assert_eq!(i.len(), 1);
		&mut self.x[i[0]]
	}
	
	fn get_rank(&self) -> Vec<IndexType> {
		vec![IndexType::Contravariant]
	}
			
}
	
impl<T> Index<usize> for Vector<T> where T: CoordinateSystem {
	type Output = f64;
	
	fn index(&self, idx: usize) -> &f64 {
		self.get_coord(&[idx])
	}
}

impl<T> IndexMut<usize> for Vector<T> where T: CoordinateSystem {
	fn index_mut(&mut self, idx: usize) -> &mut f64 {
		self.get_coord_mut(&[idx])
	}
}
	
impl<T> Vector<T> where T: CoordinateSystem {
	
	/// Creates a new vector with origin at `origin` and coordinates `coords`
	pub fn new_at_point(origin: Point<T>, coords: &[f64]) -> Vector<T> {
		Vector {
			p: origin,
			x: Vec::from(coords)
		}
	}
	
	/// Creates a new vector with origin at `origin` and coordinates `coords`
	pub fn new(origin: &[f64], coords: &[f64]) -> Vector<T> {
		Vector::new_at_point(
			Point::new(origin),
			coords
		)
	}
	
}