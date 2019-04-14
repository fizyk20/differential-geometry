use super::coordinates::{CoordinateSystem, Point};
use super::tensors::{
    ContravariantIndex, CovariantIndex, InnerProduct, InvTwoForm, Tensor, TwoForm,
};
use crate::inner;
use crate::typenum::consts::{U0, U1, U2, U3};
use crate::typenum::{Exp, Pow, Unsigned};
use generic_array::ArrayLength;

/// Trait representing the metric properties of the coordinate system
pub trait MetricSystem: CoordinateSystem
where
    <Self as CoordinateSystem>::Dimension: Pow<U2> + Pow<U3>,
    Exp<<Self as CoordinateSystem>::Dimension, U2>: ArrayLength<f64>,
    Exp<<Self as CoordinateSystem>::Dimension, U3>: ArrayLength<f64>,
{
    /// Returns the metric tensor at a given point.
    fn g(point: &Point<Self>) -> TwoForm<Self>;

    /// Returns the inverse metric tensor at a given point.
    ///
    /// The default implementation calculates the metric and then inverts it. A direct
    /// implementation may be desirable for more performance.
    fn inv_g(point: &Point<Self>) -> InvTwoForm<Self> {
        Self::g(point).inverse().unwrap()
    }

    /// Returns the partial derivatives of the metric at a given point.
    ///
    /// The default implementation calculates them numerically. A direct implementation
    /// may be desirable for performance.
    fn dg(point: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))> {
        let d = Self::dimension();
        let mut result = Tensor::zero(point.clone());
        let h = Self::small(point);

        for j in 0..d {
            let mut x = point.clone();
            x[j] = x[j] - h;
            let g1 = Self::g(&x);

            x[j] = x[j] + h * 2.0;
            let g2 = Self::g(&x);

            for coord in g1.iter_coords() {
                // calculate dg_i/dx^j
                let index = [coord[0], coord[1], j];
                result[&index[..]] = (g2[&*coord] - g1[&*coord]) / (2.0 * h);
            }
        }

        result
    }

    /// Returns the covariant Christoffel symbols (with three lower indices).
    ///
    /// The default implementation calculates them from the metric. A direct implementation
    /// may be desirable for performance.
    fn covariant_christoffel(
        point: &Point<Self>,
    ) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))> {
        let dg = Self::dg(point);
        let mut result =
            Tensor::<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>::zero(point.clone());

        for i in result.iter_coords() {
            result[&*i] =
                0.5 * (dg[&*i] + dg[&[i[0], i[2], i[1]][..]] - dg[&[i[1], i[2], i[0]][..]]);
        }

        result
    }

    /// Returns the Christoffel symbols.
    ///
    /// The default implementation calculates them from the metric. A direct implementation
    /// may be desirable for performance.
    fn christoffel(
        point: &Point<Self>,
    ) -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))> {
        let ig = Self::inv_g(point);
        let gamma = Self::covariant_christoffel(point);

        <InvTwoForm<Self> as InnerProduct<
            Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>,
            U1,
            U2,
        >>::inner_product(ig, gamma)
    }
}

impl<T> Tensor<T, ContravariantIndex>
where
    T: MetricSystem,
    T::Dimension: Pow<U1> + Pow<U2> + Pow<U3> + Unsigned,
    Exp<T::Dimension, U1>: ArrayLength<f64>,
    Exp<T::Dimension, U2>: ArrayLength<f64>,
    Exp<T::Dimension, U3>: ArrayLength<f64>,
{
    pub fn square(&self) -> f64 {
        let g = T::g(self.get_point());
        let temp = inner!(_, _; U1, U2; g, self.clone());
        *inner!(_, _; U0, U1; temp, self.clone())
    }

    pub fn normalize(&mut self) {
        let len = self.square().abs().sqrt();
        for i in 0..T::Dimension::to_usize() {
            self[i] /= len;
        }
    }
}

impl<T> Tensor<T, CovariantIndex>
where
    T: MetricSystem,
    T::Dimension: Pow<U1> + Pow<U2> + Pow<U3> + Unsigned,
    Exp<T::Dimension, U1>: ArrayLength<f64>,
    Exp<T::Dimension, U2>: ArrayLength<f64>,
    Exp<T::Dimension, U3>: ArrayLength<f64>,
{
    pub fn square(&self) -> f64 {
        let g = T::inv_g(self.get_point());
        let temp = inner!(_, _; U1, U2; g, self.clone());
        *inner!(_, _; U0, U1; temp, self.clone())
    }

    pub fn normalize(&mut self) {
        let len = self.square().abs().sqrt();
        for i in 0..T::Dimension::to_usize() {
            self[i] /= len;
        }
    }
}
