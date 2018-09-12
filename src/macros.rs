#[macro_export]
macro_rules! mul {
    ($T1: ty, $T2: ty; $op1: expr, $op2: expr) => {{
        use std::ops::Mul;
        <$T1 as Mul<$T2>>::mul($op1, $op2)
    }};
}

#[macro_export]
macro_rules! inner {
    ($T1: ty, $T2: ty; $I1: ty, $I2: ty; $op1: expr, $op2: expr) => {{
        use $crate::tensors::InnerProduct;
        <$T1 as InnerProduct<$T2, $I1, $I2>>::inner_product($op1, $op2)
    }};
}
