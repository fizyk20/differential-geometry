use super::coordinates::{Point, CoordinateSystem};
use super::tensors::{TwoForm, InvTwoForm, Tensor, CovariantIndex, ContravariantIndex, InnerProduct};
use typenum::{Pow, Exp};
use typenum::consts::{U1, U2, U3};
use generic_array::ArrayLength;

pub trait MetricSystem: CoordinateSystem
	where <Self as CoordinateSystem>::Dimension: Pow<U2> + Pow<U3>,
	      Exp<<Self as CoordinateSystem>::Dimension, U2>: ArrayLength<f64>,
	      Exp<<Self as CoordinateSystem>::Dimension, U3>: ArrayLength<f64>
{
    fn g(point: &Point<Self>) -> TwoForm<Self>;
    fn inv_g(point: &Point<Self>) -> InvTwoForm<Self> { Self::g(point).inverse().unwrap() }
    
    fn dg(point: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))> {
    	let d = Self::dimension();
        let mut result = Tensor::new(point.clone());
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
    
    fn covariant_christoffel(point: &Point<Self>) -> Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))> {
    	let dg = Self::dg(point);
    	let mut result = Tensor::<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>::new(point.clone());
    	
    	for i in result.iter_coords() {
    	    result[&*i] = 0.5*(dg[&*i] + dg[&[i[0],i[2],i[1]][..]] - dg[&[i[1], i[2], i[0]][..]]);
    	}
    	
    	result
    }

    fn christoffel(point: &Point<Self>) -> Tensor<Self, (ContravariantIndex, (CovariantIndex, CovariantIndex))> {
        let ig = Self::inv_g(point);
        let gamma = Self::covariant_christoffel(point);
        
        <InvTwoForm<Self> as InnerProduct<Tensor<Self, (CovariantIndex, (CovariantIndex, CovariantIndex))>, U1, U2>>::inner_product(ig, gamma)
    }
}
