use typenum::consts::{U0, U1, U2, U4};
use coordinates::{CoordinateSystem, Point};
use tensors::{Vector, Covector, Matrix, Tensor, ContravariantIndex, InnerProduct};
use generic_array::GenericArray;

struct Test2;
impl CoordinateSystem for Test2 {
    type Dimension = U2;
}

struct Test4;
impl CoordinateSystem for Test4 {
    type Dimension = U4;
}

#[test]
fn test_ranks() {
    assert_eq!(Vector::<Test4>::get_rank(), 1);
    assert_eq!(Matrix::<Test4>::get_rank(), 2);
}

#[test]
fn test_num_coords() {
    assert_eq!(Vector::<Test4>::get_num_coords(), 4);
    assert_eq!(Matrix::<Test4>::get_num_coords(), 16);
}

#[test]
fn test_iter_coords() {
    let p1 = Point::new(GenericArray::new());
    let matrix1 = Matrix::<Test2>::new(p1);
    let p2 = Point::new(GenericArray::new());
    let matrix2 = Matrix::<Test4>::new(p2);

    let mut i = 0;
    for _ in matrix1.iter_coords() {
        i += 1;
    }
    assert_eq!(i, 4);

    i = 0;
    for _ in matrix2.iter_coords() {
        i += 1;
    }
    assert_eq!(i, 16);
}

#[test]
fn test_add() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);
    let mut vector2 = Vector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    vector2[0] = 1.5;
    vector2[1] = 1.6;

    let result = vector1 + vector2;

    assert_eq!(result[0], 2.5);
    assert_eq!(result[1], 3.6);
}

#[test]
fn test_sub() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);
    let mut vector2 = Vector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    vector2[0] = 1.5;
    vector2[1] = 1.75;

    let result = vector1 - vector2;

    assert_eq!(result[0], -0.5);
    assert_eq!(result[1], 0.25);
}

#[test]
fn test_trace() {
    let p = Point::new(GenericArray::new());
    let mut matrix = Matrix::<Test2>::new(p);

    matrix[0] = 1.0;
    matrix[1] = 3.0;
    matrix[2] = 0.0;
    matrix[3] = 3.0;

    let tr = matrix.trace::<U0, U1>();

    assert_eq!(*tr, 4.0);
}

// needed for tests below
use std::ops::Mul;

#[test]
fn test_mul_trait() {
    assert_eq!(<Vector<Test2> as Mul<Vector<Test2>>>::Output::get_rank(), 2);
    assert_eq!(<Vector<Test2> as Mul<Vector<Test2>>>::Output::get_num_coords(), 4);
    assert_eq!(<Vector<Test2> as Mul<f64>>::Output::get_rank(), 1);
    assert_eq!(<Vector<Test2> as Mul<f64>>::Output::get_num_coords(), 2);
}

#[test]
fn test_mul_scalar() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    // this works
    let result: Vector<Test2> = <Vector<Test2> as Mul<f64>>::mul(vector1, 5.0);
    // this doesn't
    // let result: Vector<Test2> = vector1 * 5.0;

    assert_eq!(result[0], 5.0);
    assert_eq!(result[1], 10.0);
}

#[test]
fn test_mul_vector() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);
    let mut vector2 = Vector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    vector2[0] = 3.0;
    vector2[1] = 4.0;

    // this works
    let result: Tensor<Test2, (ContravariantIndex, ContravariantIndex)> = <Vector<Test2> as Mul<Vector<Test2>>>::mul(vector1, vector2);
    // this doesn't
    // let result = vector1 * vector2;

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], 6.0);
    assert_eq!(result[3], 8.0);
}

#[test]
fn test_inner_product() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);
    let mut vector2 = Covector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    vector2[0] = 3.0;
    vector2[1] = 4.0;

    let result: Tensor<Test2, ()> = <Vector<Test2> as InnerProduct<Covector<Test2>, U0, U1>>::inner_product(vector1, vector2);

    assert_eq!(*result, 11.0);
}
