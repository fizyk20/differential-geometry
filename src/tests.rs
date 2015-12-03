use typenum::consts::U4;
use coordinates::CoordinateSystem;
use tensors::{Vector, Matrix};

struct Test;
impl CoordinateSystem for Test {
    type Dimension = U4;
}

#[test]
fn test_ranks() {
    assert_eq!(Vector::<Test>::get_rank(), 1);
    assert_eq!(Matrix::<Test>::get_rank(), 2);
}

#[test]
fn test_num_coords() {
    assert_eq!(Vector::<Test>::get_num_coords(), 4);
    assert_eq!(Matrix::<Test>::get_num_coords(), 16);
}
