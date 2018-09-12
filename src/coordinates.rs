//! Module containing basic types representing coordinate systems.

use super::tensors::{ContravariantIndex, CovariantIndex, Matrix, Tensor};
use generic_array::{ArrayLength, GenericArray};
use std::ops::{Index, IndexMut};
use typenum::consts::U2;
use typenum::uint::Unsigned;
use typenum::Pow;

/// `CoordinateSystem` marks a struct (usually a unit struct) as representing a coordinate system.
pub trait CoordinateSystem: Sized {
    /// An associated type representing the dimension of the coordinate system
    type Dimension: Unsigned + ArrayLength<f64> + ArrayLength<usize>;

    /// Function returning a small value for purposes of numerical differentiation.
    /// What is considered a small value may depend on the point, hence the parameter.
    /// Returns just 0.01 by default.
    fn small(_: &Point<Self>) -> f64 {
        0.01
    }

    /// Function returning the dimension
    fn dimension() -> usize {
        Self::Dimension::to_usize()
    }
}

/// Struct representing a point on the manifold. The information about the coordinate system
/// is saved in the type parameter, so that only operations on objects belonging to the same
/// coordinate system will be allowed.
pub struct Point<T: CoordinateSystem> {
    /// The coordinates of the point.
    x: GenericArray<f64, T::Dimension>,
}

impl<T> Point<T>
where
    T: CoordinateSystem,
{
    /// Creates a new point with coordinates described by the array
    pub fn new(coords: GenericArray<f64, T::Dimension>) -> Point<T> {
        Point { x: coords }
    }

    /// Creates a new point with coordinates passed in the slice
    pub fn from_slice(coords: &[f64]) -> Point<T> {
        Point {
            x: GenericArray::clone_from_slice(coords),
        }
    }
}

impl<T> Clone for Point<T>
where
    T: CoordinateSystem,
{
    fn clone(&self) -> Point<T> {
        Point::new(self.x.clone())
    }
}

impl<T> Copy for Point<T>
where
    T: CoordinateSystem,
    <T::Dimension as ArrayLength<f64>>::ArrayType: Copy,
{}

impl<T> Index<usize> for Point<T>
where
    T: CoordinateSystem,
{
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        &self.x[idx]
    }
}

impl<T> IndexMut<usize> for Point<T>
where
    T: CoordinateSystem,
{
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.x[idx]
    }
}

impl<T> PartialEq<Point<T>> for Point<T>
where
    T: CoordinateSystem,
{
    fn eq(&self, rhs: &Point<T>) -> bool {
        (0..T::dimension()).all(|i| self[i] == rhs[i])
    }
}

impl<T> Eq for Point<T> where T: CoordinateSystem {}

/// Trait used for conversions between different coordinate systems. Implementing `ConversionTo<T>`
/// for a `CoordinateSystem` will allow objects in that system to be converted to the system `T`
/// (note that `T` also has to be a `CoordinateSystem`).
pub trait ConversionTo<T: CoordinateSystem + 'static>: CoordinateSystem
where
    T::Dimension: Pow<U2>,
    <T::Dimension as Pow<U2>>::Output: ArrayLength<f64>,
{
    /// Function converting the coordinates of a point.
    fn convert_point(p: &Point<Self>) -> Point<T>;

    /// Function calculating a Jacobian at a point - that is, the matrix of derivatives
    /// of the coordinate conversions.
    ///
    /// This will be contracted with contravariant indices in the tensor.
    fn jacobian(p: &Point<Self>) -> Matrix<T> {
        let d = Self::dimension();
        let mut result = Matrix::zero(Self::convert_point(p));
        let h = Self::small(p);

        for j in 0..d {
            let mut x = p.clone();
            x[j] = x[j] - h;
            let y1 = Self::convert_point(&x);

            x[j] = x[j] + h * 2.0;
            let y2 = Self::convert_point(&x);

            for i in 0..d {
                // calculate dyi/dxj
                let index = [i, j];
                result[&index[..]] = (y2[i] - y1[i]) / (2.0 * h);
            }
        }

        result
    }

    /// The inverse matrix of the Jacobian at a point.
    ///
    /// In conversions, it will be contracted with covariant indices.
    fn inv_jacobian(p: &Point<Self>) -> Tensor<T, (CovariantIndex, ContravariantIndex)> {
        ConversionTo::<T>::jacobian(p).inverse().unwrap()
    }
}
