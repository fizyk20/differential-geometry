use typenum::consts::{U2, U4};
use coordinates::{CoordinateSystem, Point};
use tensors::{Vector, Matrix};
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
fn test_mul_scalar() {
    let p = Point::new(GenericArray::new());
    let mut vector1 = Vector::<Test2>::new(p);

    vector1[0] = 1.0;
    vector1[1] = 2.0;

    let result = vector1 * 5.0;

    assert_eq!(result[0], 5.0);
    assert_eq!(result[1], 10.0);
}
