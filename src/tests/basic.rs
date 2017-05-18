use typenum::consts::{U0, U1, U2, U4};
use coordinates::{CoordinateSystem, Point};
use tensors::{Vector, Covector, Matrix, TwoForm, InvTwoForm, Scalar};
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
    let p1 = Point::new(GenericArray::default());
    let matrix1 = Matrix::<Test2>::zero(p1);
    let p2 = Point::new(GenericArray::default());
    let matrix2 = Matrix::<Test4>::zero(p2);

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
    let p = Point::new(GenericArray::default());
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    let vector2 = Vector::<Test2>::new(p, arr![f64; 1.5, 1.6]);

    let result = vector1 + vector2;

    assert_eq!(result[0], 2.5);
    assert_eq!(result[1], 3.6);
}

#[test]
fn test_sub() {
    let p = Point::new(GenericArray::default());
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    let vector2 = Vector::<Test2>::new(p, arr![f64; 1.5, 1.75]);

    let result = vector1 - vector2;

    assert_eq!(result[0], -0.5);
    assert_eq!(result[1], 0.25);
}

#[test]
fn test_trace() {
    let p = Point::new(GenericArray::default());
    let matrix = Matrix::<Test2>::new(p, arr![f64; 1.0, 3.0, 0.0, 3.0]);

    let tr = matrix.trace::<U0, U1>();

    assert_eq!(*tr, 4.0);
}

// needed for tests below
use std::ops::Mul;

#[test]
fn test_mul_trait() {
    assert_eq!(<Vector<Test2> as Mul<Vector<Test2>>>::Output::get_rank(), 2);
    assert_eq!(<Vector<Test2> as Mul<Vector<Test2>>>::Output::get_num_coords(),
               4);
    assert_eq!(<Vector<Test2> as Mul<f64>>::Output::get_rank(), 1);
    assert_eq!(<Vector<Test2> as Mul<f64>>::Output::get_num_coords(), 2);
}

#[test]
fn test_mul_scalar() {
    let p = Point::new(GenericArray::default());
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    // this works
    let result: Vector<Test2> = mul!(_, f64; vector1, 5.0);
    // this doesn't
    // let result: Vector<Test2> = vector1 * 5.0;

    assert_eq!(result[0], 5.0);
    assert_eq!(result[1], 10.0);
}

#[test]
fn test_mul_vector() {
    let p = Point::new(GenericArray::default());
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    let vector2 = Vector::<Test2>::new(p, arr![f64; 3.0, 4.0]);

    // this works
    let result: InvTwoForm<Test2> = mul!(_, Vector<Test2>; vector1, vector2);
    // this doesn't
    // let result = vector1 * vector2;

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 4.0);
    assert_eq!(result[2], 6.0);
    assert_eq!(result[3], 8.0);
}

#[test]
fn test_inner_product() {
    let p = Point::new(GenericArray::default());
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    let vector2 = Covector::<Test2>::new(p, arr![f64; 3.0, 4.0]);

    let result: Scalar<Test2> = inner!(_, Covector<Test2>; U0, U1; vector1, vector2);

    assert_eq!(*result, 11.0);
}

#[test]
fn test_complex_inner_product() {
    let p = Point::new(GenericArray::default());
    let form = TwoForm::<Test2>::new(p, arr![f64; 1.0, 0.0, 0.0, 1.0]);
    let vector1 = Vector::<Test2>::new(p, arr![f64; 1.0, 2.0]);
    let vector2 = Vector::<Test2>::new(p, arr![f64; 3.0, 4.0]);

    let temp = inner!(_, Vector<Test2>; U0, U2; form, vector1);
    let result = inner!(_, Vector<Test2>; U0, U1; temp, vector2);

    assert_eq!(*result, 11.0);
}

#[test]
fn test_transpose() {
    let p = Point::new(GenericArray::default());
    let matrix = Matrix::<Test2>::new(p, arr![f64; 1.0, 2.0, 3.0, 4.0]);

    let result = matrix.transpose();

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], 2.0);
    assert_eq!(result[3], 4.0);
}

#[test]
fn test_inverse() {
    let p = Point::new(GenericArray::default());
    let matrix = Matrix::<Test2>::new(p, arr![f64; 1.0, 2.0, 3.0, 4.0]);

    let result = matrix.inverse().unwrap();

    // unfortunately the inverse matrix calculation isn't perfectly accurate
    let epsilon = 0.0000001;

    assert!((result[0] + 2.0).abs() < epsilon);
    assert!((result[1] - 1.0).abs() < epsilon);
    assert!((result[2] - 1.5).abs() < epsilon);
    assert!((result[3] + 0.5).abs() < epsilon);
}
