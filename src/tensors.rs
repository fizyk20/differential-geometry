use super::coordinates::{CoordinateSystem, Point};

pub enum IndexType {
	Covariant,
	Contravariant
}

pub trait Tensor<T: CoordinateSystem> {
	
	fn get_point(&self) -> &Point<T>;
	fn get_coord(&self, &[u8]) -> T::CoordType;
	fn get_rank(&self) -> Vec<IndexType>;
	
}