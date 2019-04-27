use crate::coordinates::{ConversionTo, CoordinateSystem, Point};
use crate::tensors::Vector;
use crate::typenum::consts::U3;
use generic_array::arr;

struct Cartesian;
struct Spherical;

impl CoordinateSystem for Cartesian {
    type Dimension = U3;
}

impl CoordinateSystem for Spherical {
    type Dimension = U3;
}

impl ConversionTo<Spherical> for Cartesian {
    fn convert_point(p: &Point<Cartesian>) -> Point<Spherical> {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let theta = (p[2] / r).acos();
        let phi = p[1].atan2(p[0]);
        Point::new(arr![f64; r, theta, phi])
    }
}

#[test]
fn test_vector_to_spherical() {
    let p = Point::new(arr![f64; 0.0, 1.0, 1.0]);
    let v = Vector::<Cartesian>::new(p, arr![f64; 0.0, 0.0, 1.0]);
    let v2: Vector<Spherical> = v.convert();
    assert!((v2[0] - 0.5_f64.sqrt()).abs() < 0.00001);
    assert!((v2[1] + 0.5).abs() < 0.00001);
    assert_eq!(v2[2], 0.0);
}
