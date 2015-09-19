use super::coordinates::{CoordinateSystem, Point};

pub enum IndexType {
	Covariant,
	Contravariant
}

pub struct Tensor<T: CoordinateSystem> {
	rank: Vec<IndexType>,
	point: Point<T>,
	coords:	Vec<T::CoordType>
}