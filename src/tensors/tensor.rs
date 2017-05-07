//! This module defines the `Tensor` type and all sorts of operations on it.
use coordinates::{CoordinateSystem, Point, ConversionTo};
use std::ops::{Index, IndexMut};
use std::ops::{Add, Sub, Mul, Div, Deref, DerefMut};
use typenum::uint::Unsigned;
use typenum::consts::{U1, U2};
use typenum::{Pow, Same};
use generic_array::{GenericArray, ArrayLength};
use super::{CovariantIndex, ContravariantIndex, TensorIndex, Variance, IndexType};
use super::variance::{Concat, Contract, Joined, Contracted, Add1, OtherIndex};

/// Helper type for `typenum` powers
pub type Power<T, U> = <T as Pow<U>>::Output;

/// Struct representing a tensor.
///
/// A tensor is anchored at a given point and has coordinates
/// represented in the system defined by the generic parameter
/// `T`. The variance of the tensor (meaning its rank and types
/// of its indices) is defined by `V`. This allows Rust
/// to decide at compile time whether two tensors are legal
/// to be added / multiplied / etc.
///
/// It is only OK to perform an operation on two tensors if
/// they belong to the same coordinate system.
pub struct Tensor<T: CoordinateSystem, U: Variance>
    where T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    p: Point<T>,
    x: GenericArray<f64, Power<T::Dimension, U::Rank>>,
}

impl<T, U> Clone for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    fn clone(&self) -> Tensor<T, U> {
        Tensor {
            p: self.p.clone(),
            x: self.x.clone(),
        }
    }
}

impl<T, U> Copy for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          <T::Dimension as ArrayLength<f64>>::ArrayType: Copy,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>,
          <Power<T::Dimension, U::Rank> as ArrayLength<f64>>::ArrayType: Copy
{
}

/// A struct for iterating over the coordinates of a tensor.
pub struct CoordIterator<U>
    where U: Variance,
          U::Rank: ArrayLength<usize>
{
    started: bool,
    dimension: usize,
    cur_coord: GenericArray<usize, U::Rank>,
}

impl<U> CoordIterator<U>
    where U: Variance,
          U::Rank: ArrayLength<usize>
{
    pub fn new(dimension: usize) -> CoordIterator<U> {
        CoordIterator {
            started: false,
            dimension: dimension,
            cur_coord: GenericArray::default(),
        }
    }
}

impl<U> Iterator for CoordIterator<U>
    where U: Variance,
          U::Rank: ArrayLength<usize>
{
    type Item = GenericArray<usize, U::Rank>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            return Some(self.cur_coord.clone());
        }

        // handle scalars
        if self.cur_coord.len() < 1 {
            return None;
        }

        let mut i = self.cur_coord.len() - 1;
        loop {
            self.cur_coord[i] += 1;
            if self.cur_coord[i] < self.dimension {
                break;
            }
            self.cur_coord[i] = 0;
            if i == 0 {
                return None;
            }
            i -= 1;
        }

        Some(self.cur_coord.clone())
    }
}

impl<T, V> Tensor<T, V>
    where T: CoordinateSystem,
          V: Variance,
          T::Dimension: Pow<V::Rank>,
          Power<T::Dimension, V::Rank>: ArrayLength<f64>
{
    /// Returns the point at which the tensor is defined.
    pub fn get_point(&self) -> &Point<T> {
        &self.p
    }

    /// Converts a set of tensor indices passed as a slice into a single index for the internal array.
    ///
    /// The length of the slice (the number of indices) has to be compatible with the rank of the tensor.
    pub fn get_coord(i: &[usize]) -> usize {
        assert_eq!(i.len(), V::rank());
        let dim = T::dimension();
        let index = i.into_iter()
            .fold(0, |res, idx| {
                assert!(*idx < dim);
                res * dim + idx
            });
        index
    }

    /// Returns the variance of the tensor, that is, the list of the index types.
    /// A vector would return vec![Contravariant], a metric tensor: vec![Covariant, Covariant].
    pub fn get_variance() -> Vec<IndexType> {
        V::variance()
    }

    /// Returns the rank of the tensor
    pub fn get_rank() -> usize {
        V::rank()
    }

    /// Returns the number of coordinates of the tensor (equal to [Dimension]^[Rank])
    pub fn get_num_coords() -> usize {
        <T::Dimension as Pow<V::Rank>>::Output::to_usize()
    }

    /// Creates a new, zero tensor at a given point
    pub fn zero(point: Point<T>) -> Tensor<T, V> {
        Tensor {
            p: point,
            x: GenericArray::default(),
        }
    }

    /// Creates a tensor at a given point with the coordinates defined by the array.
    ///
    /// The number of elements in the array must be equal to the number of coordinates
    /// of the tensor.
    ///
    /// One-dimensional array represents an n-dimensional tensor in such a way, that
    /// the last index is the one that is changing the most often, i.e. the sequence is
    /// as follows: (0,0,...,0), (0,0,...,1), (0,0,...,2), ..., (0,0,...,1,0), (0,0,...,1,1), ... etc.
    pub fn new(point: Point<T>,
               coords: GenericArray<f64, Power<T::Dimension, V::Rank>>)
               -> Tensor<T, V> {
        Tensor {
            p: point,
            x: coords,
        }
    }

    /// Creates a tensor at a given point with the coordinates defined by the slice.
    ///
    /// The number of elements in the slice must be equal to the number of coordinates
    /// of the tensor.
    ///
    /// One-dimensional slice represents an n-dimensional tensor in such a way, that
    /// the last index is the one that is changing the most often, i.e. the sequence is
    /// as follows: (0,0,...,0), (0,0,...,1), (0,0,...,2), ..., (0,0,...,1,0), (0,0,...,1,1), ... etc.
    pub fn from_slice(point: Point<T>, slice: &[f64]) -> Tensor<T, V> {
        assert_eq!(Tensor::<T, V>::get_num_coords(), slice.len());
        Tensor {
            p: point,
            x: GenericArray::clone_from_slice(slice),
        }
    }

    /// Contracts two indices
    ///
    /// The indices must be of opposite types. This is checked at compile time.
    pub fn trace<Ul, Uh>(&self) -> Tensor<T, Contracted<V, Ul, Uh>>
        where Ul: Unsigned,
              Uh: Unsigned,
              V: Contract<Ul, Uh>,
              <Contracted<V, Ul, Uh> as Variance>::Rank: ArrayLength<usize>,
              T::Dimension: Pow<<Contracted<V, Ul, Uh> as Variance>::Rank>,
              Power<T::Dimension, <Contracted<V, Ul, Uh> as Variance>::Rank>: ArrayLength<f64>
    {
        let index1 = Ul::to_usize();
        let index2 = Uh::to_usize();

        let mut result = Tensor::<T, Contracted<V, Ul, Uh>>::zero(self.p.clone());

        for coord in result.iter_coords() {
            let mut sum = 0.0;

            for i in 0..T::dimension() {
                let mut vec_coords = coord.to_vec();
                vec_coords.insert(index1, i);
                vec_coords.insert(index2, i);
                sum += self[&*vec_coords];
            }

            result[&*coord] = sum;
        }

        result
    }
}

impl<T, U> Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          U::Rank: ArrayLength<usize>,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    /// Returns an iterator over the coordinates of the tensor.
    pub fn iter_coords(&self) -> CoordIterator<U> {
        CoordIterator::new(T::dimension())
    }
}

impl<'a, T, U> Index<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = f64;

    fn index(&self, idx: &'a [usize]) -> &f64 {
        &self.x[Self::get_coord(idx)]
    }
}

impl<'a, T, U> IndexMut<&'a [usize]> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    fn index_mut(&mut self, idx: &'a [usize]) -> &mut f64 {
        &mut self.x[Self::get_coord(idx)]
    }
}

impl<'a, T, U> Index<usize> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        &self.x[idx]
    }
}

impl<'a, T, U> IndexMut<usize> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    fn index_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self.x[idx]
    }
}

/// A scalar type, which is a tensor with rank 0.
///
/// This is de facto just a number, so it implements `Deref` and `DerefMut` into `f64`.
pub type Scalar<T> = Tensor<T, ()>;

/// A vector type (rank 1 contravariant tensor)
pub type Vector<T> = Tensor<T, ContravariantIndex>;

/// A covector type (rank 1 covariant tensor)
pub type Covector<T> = Tensor<T, CovariantIndex>;

/// A matrix type (rank 2 contravariant-covariant tensor)
pub type Matrix<T> = Tensor<T, (ContravariantIndex, CovariantIndex)>;

/// A bilinear form type (rank 2 doubly covariant tensor)
pub type TwoForm<T> = Tensor<T, (CovariantIndex, CovariantIndex)>;

/// A rank 2 doubly contravariant tensor
pub type InvTwoForm<T> = Tensor<T, (ContravariantIndex, ContravariantIndex)>;

impl<T: CoordinateSystem> Deref for Scalar<T> {
    type Target = f64;

    fn deref(&self) -> &f64 {
        &self.x[0]
    }
}

impl<T: CoordinateSystem> DerefMut for Scalar<T> {
    fn deref_mut(&mut self) -> &mut f64 {
        &mut self.x[0]
    }
}

// Arithmetic operations

impl<T, U> Add<Tensor<T, U>> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, U>;

    fn add(mut self, rhs: Tensor<T, U>) -> Tensor<T, U> {
        assert!(self.p == rhs.p);
        for i in 0..(Tensor::<T, U>::get_num_coords()) {
            self[i] = self[i] + rhs[i];
        }
        self
    }
}

impl<T, U> Sub<Tensor<T, U>> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, U>;

    fn sub(mut self, rhs: Tensor<T, U>) -> Tensor<T, U> {
        assert!(self.p == rhs.p);
        for i in 0..(Tensor::<T, U>::get_num_coords()) {
            self[i] = self[i] - rhs[i];
        }
        self
    }
}

impl<T, U> Mul<f64> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, U>;

    fn mul(mut self, rhs: f64) -> Tensor<T, U> {
        for i in 0..(Tensor::<T, U>::get_num_coords()) {
            self[i] = self[i] * rhs;
        }
        self
    }
}

impl<T, U> Mul<Tensor<T, U>> for f64
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, U>;

    fn mul(self, mut rhs: Tensor<T, U>) -> Tensor<T, U> {
        for i in 0..(Tensor::<T, U>::get_num_coords()) {
            rhs[i] = rhs[i] * self;
        }
        rhs
    }
}

impl<T, U> Div<f64> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, U>;

    fn div(mut self, rhs: f64) -> Tensor<T, U> {
        for i in 0..(Tensor::<T, U>::get_num_coords()) {
            self[i] = self[i] / rhs;
        }
        self
    }
}

// Tensor multiplication

// For some reason this triggers recursion overflow when tested - to be investigated
impl<T, U, V> Mul<Tensor<T, V>> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          V: Variance,
          U::Rank: ArrayLength<usize>,
          V::Rank: ArrayLength<usize>,
          T::Dimension: Pow<U::Rank> + Pow<V::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>,
          Power<T::Dimension, V::Rank>: ArrayLength<f64>,
          U: Concat<V>,
          Joined<U, V>: Variance,
          T::Dimension: Pow<<Joined<U, V> as Variance>::Rank>,
          Power<T::Dimension, <Joined<U, V> as Variance>::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, Joined<U, V>>;

    fn mul(self, rhs: Tensor<T, V>) -> Tensor<T, Joined<U, V>> {
        assert!(self.p == rhs.p);
        let mut result = Tensor::zero(self.p.clone());
        for coord1 in self.iter_coords() {
            for coord2 in rhs.iter_coords() {
                let mut vec_coord1 = coord1.to_vec();
                let mut vec_coord2 = coord2.to_vec();
                vec_coord1.append(&mut vec_coord2);
                let index: &[usize] = &vec_coord1;
                let index1: &[usize] = &coord1;
                let index2: &[usize] = &coord2;
                result[index] = self[index1] * rhs[index2];
            }
        }
        result
    }
}

/// Trait representing the inner product of two tensors.
///
/// The inner product is just a multiplication followed by a contraction.
/// The contraction is defined by type parameters `Ul` and `Uh`. `Ul` has to
/// be less than `Uh` and the indices at those positions must be of opposite types
/// (checked at compile time)
pub trait InnerProduct<Rhs, Ul: Unsigned, Uh: Unsigned> {
    type Output;

    fn inner_product(self, rhs: Rhs) -> Self::Output;
}

impl<T, U, V, Ul, Uh> InnerProduct<Tensor<T, V>, Ul, Uh> for Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          V: Variance,
          Ul: Unsigned,
          Uh: Unsigned,
          T::Dimension: Pow<U::Rank> + Pow<V::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>,
          Power<T::Dimension, V::Rank>: ArrayLength<f64>,
          U: Concat<V>,
          Joined<U,V>: Contract<Ul, Uh>,
          <Contracted<Joined<U, V>, Ul, Uh> as Variance>::Rank: ArrayLength<usize>,
          T::Dimension: Pow<<Contracted<Joined<U, V>, Ul, Uh> as Variance>::Rank>,
          Power<T::Dimension, <Contracted<Joined<U, V>, Ul, Uh> as Variance>::Rank>: ArrayLength<f64>
{
    type Output = Tensor<T, Contracted<Joined<U, V>, Ul, Uh>>;

    fn inner_product(self, rhs: Tensor<T, V>) -> Tensor<T, Contracted<Joined<U, V>, Ul, Uh>> {
        assert!(self.p == rhs.p);
        let mut result = Tensor::<T, Contracted<Joined<U, V>, Ul, Uh>>::zero(self.p.clone());

        for coord_res in result.iter_coords() {
            let mut sum = 0.0;
            for i in 0..T::dimension() {
                let mut coords = coord_res.to_vec();
                coords.insert(Ul::to_usize(), i);
                coords.insert(Uh::to_usize(), i);
                let (coords1, coords2) = coords.split_at(U::Rank::to_usize());
                sum += self[coords1]*rhs[coords2];
            }
            result[&*coord_res] = sum;
        }

        result
    }
}


impl<T, Ul, Ur> Tensor<T, (Ul, Ur)>
    where T: CoordinateSystem,
          Ul: TensorIndex + OtherIndex,
          Ur: TensorIndex + OtherIndex,
          Add1<Ul::Rank>: Unsigned + Add<U1>,
          Add1<Ur::Rank>: Unsigned + Add<U1>,
          Add1<<<Ul as OtherIndex>::Output as Variance>::Rank>: Unsigned + Add<U1>,
          Add1<<<Ur as OtherIndex>::Output as Variance>::Rank>: Unsigned + Add<U1>,
          <(Ul, Ur) as Variance>::Rank: ArrayLength<usize>,
          T::Dimension: Pow<Add1<Ul::Rank>> + Pow<Add1<Ur::Rank>> + ArrayLength<usize>,
          T::Dimension: Pow<Add1<<<Ul as OtherIndex>::Output as Variance>::Rank>>,
          T::Dimension: Pow<Add1<<<Ur as OtherIndex>::Output as Variance>::Rank>>,
          Power<T::Dimension, Add1<Ul::Rank>>: ArrayLength<f64>,
          Power<T::Dimension, Add1<Ur::Rank>>: ArrayLength<f64>,
          Power<T::Dimension, Add1<<<Ul as OtherIndex>::Output as Variance>::Rank>>: ArrayLength<f64>,
          Power<T::Dimension, Add1<<<Ur as OtherIndex>::Output as Variance>::Rank>>: ArrayLength<f64>
{
/// Returns a unit matrix (1 on the diagonal, 0 everywhere else)
    pub fn unit(p: Point<T>) -> Tensor<T, (Ul, Ur)> {
        let mut result = Tensor::<T, (Ul, Ur)>::zero(p);

        for i in 0..T::dimension() {
            let coords: &[usize] = &[i,i];
            result[coords] = 1.0;
        }

        result
    }

/// Transposes the matrix
    pub fn transpose(&self) -> Tensor<T, (Ur, Ul)> {
        let mut result = Tensor::<T, (Ur, Ul)>::zero(self.p.clone());

        for coords in self.iter_coords() {
            let coords2: &[usize] = &[coords[1], coords[0]];
            result[coords2] = self[&*coords];
        }

        result
    }

// Function calculating the LU decomposition of a matrix - found in the internet
// The decomposition is done in-place and a permutation vector is returned (or None
// if the matrix was singular)
    fn lu_decompose(&mut self) -> Option<GenericArray<usize, T::Dimension>> {
        let n = T::dimension();
        let absmin = 1.0e-30_f64;
        let mut result = GenericArray::default();
        let mut row_norm = GenericArray::<f64, T::Dimension>::default();

        let mut max_row = 0;

        for i in 0..n {
            let mut absmax = 0.0;

            for j in 0..n {
                let coord: &[usize] = &[i,j];
                let maxtemp = self[coord].abs();
                absmax = if maxtemp > absmax { maxtemp } else { absmax };
            }

            if absmax == 0.0 {
                return None;
            }

            row_norm[i] = 1.0 / absmax;
        }

        for j in 0..n {
            for i in 0..j {
                for k in 0..i {
                    let coord1: &[usize] = &[i, j];
                    let coord2: &[usize] = &[i, k];
                    let coord3: &[usize] = &[k, j];

                    self[coord1] -= self[coord2] * self[coord3];
                }
            }

            let mut absmax = 0.0;

            for i in j..n {
                let coord1: &[usize] = &[i, j];

                for k in 0..j {
                    let coord2: &[usize] = &[i, k];
                    let coord3: &[usize] = &[k, j];

                    self[coord1] -= self[coord2] * self[coord3];
                }

                let maxtemp = self[coord1].abs() * row_norm[i];

                if maxtemp > absmax {
                    absmax = maxtemp;
                    max_row = i;
                }
            }

            if max_row != j {
                if (j == n-2) && self[&[j, j+1] as &[usize]] == 0.0 {
                    max_row = j;
                }
                else {
                    for k in 0..n {
                        let jk: &[usize] = &[j, k];
                        let maxrow_k: &[usize] = &[max_row, k];
                        let maxtemp = self[jk];
                        self[jk] = self[maxrow_k];
                        self[maxrow_k] = maxtemp;
                    }

                    row_norm[max_row] = row_norm[j];
                }
            }

            result[j] = max_row;

            let jj: &[usize] = &[j, j];

            if self[jj] == 0.0 {
                self[jj] = absmin;
            }

            if j != n-1 {
                let maxtemp = 1.0 / self[jj];
                for i in j+1..n {
                    self[&[i, j] as &[usize]] *= maxtemp;
                }
            }
        }

        Some(result)
    }

// Function solving a linear system of equations (self*x = b) using the LU decomposition
    fn lu_substitution(&self, b: &GenericArray<f64, T::Dimension>, permute: &GenericArray<usize, T::Dimension>)
        -> GenericArray<f64, T::Dimension>
    {
        let mut result = b.clone();
        let n = T::dimension();

        for i in 0..n {
            let mut tmp = result[permute[i]];
            result[permute[i]] = result[i];
            for j in (0..i).rev() {
                tmp -= self[&[i, j] as &[usize]] * result[j];
            }
            result[i] = tmp;
        }

        for i in (0..n).rev() {
            for j in i+1..n {
                result[i] -= self[&[i, j] as &[usize]] * result[j];
            }
            result[i] /= self[&[i, i] as &[usize]];
        }

        result
    }

/// Function calculating the inverse of `self` using the LU ddecomposition.
///
/// The return value is an `Option`, since `self` may be non-invertible -
/// in such a case, None is returned
    pub fn inverse(&self) -> Option<Tensor<T, (<Ul as OtherIndex>::Output, <Ur as OtherIndex>::Output)>> {
        let mut result = Tensor::<T, (<Ul as OtherIndex>::Output, <Ur as OtherIndex>::Output)>::zero(self.p.clone());

        let mut tmp = self.clone();

        let permute = match tmp.lu_decompose() {
            Some(p) => p,
            None => return None
        };

        for i in 0..T::dimension() {
            let mut dxm = GenericArray::<f64, T::Dimension>::default();
            dxm[i] = 1.0;

            let x = tmp.lu_substitution(&dxm, &permute);

            for k in 0..T::dimension() {
                result[&[k, i] as &[usize]] = x[k];
            }
        }

        Some(result)
    }
}

impl<T, U> Tensor<T, U>
    where T: CoordinateSystem,
          U: Variance,
          U::Rank: ArrayLength<usize>,
          T::Dimension: Pow<U::Rank>,
          Power<T::Dimension, U::Rank>: ArrayLength<f64>
{
    pub fn convert<T2>(&self) -> Tensor<T2, U>
        where T2: CoordinateSystem + 'static,
              T2::Dimension: Pow<U::Rank> + Pow<U2> + Same<T::Dimension>,
              Power<T2::Dimension, U::Rank>: ArrayLength<f64>,
              Power<T2::Dimension, U2>: ArrayLength<f64>,
              T: ConversionTo<T2>
    {
        let mut result = Tensor::<T2, U>::zero(<T as ConversionTo<T2>>::convert_point(&self.p));

        let jacobian = <T as ConversionTo<T2>>::jacobian(&self.p);
        let inv_jacobian = <T as ConversionTo<T2>>::inv_jacobian(&self.p);
        let variance = <U as Variance>::variance();

        for i in result.iter_coords() {
            let mut temp = 0.0;
            for j in self.iter_coords() {
                let mut temp2 = self[&*j];
                for (k, v) in variance.iter().enumerate() {
                    let coords = [i[k], j[k]];
                    temp2 *= match *v {
                        IndexType::Covariant => inv_jacobian[&coords[..]],
                        IndexType::Contravariant => jacobian[&coords[..]],
                    };
                }
                temp += temp2;
            }
            result[&*i] = temp;
        }

        result
    }
}
