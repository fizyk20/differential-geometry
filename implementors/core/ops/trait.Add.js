(function() {var implementors = {};
implementors['typenum'] = ["impl <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt; for <a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, B: <a class='trait' href='typenum/marker_traits/trait.Bit.html' title='typenum::marker_traits::Bit'>Bit</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;U, B&gt;","impl <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; for <a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;U, <a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt;","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;U, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; <span class='where'>where U: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt;, <a class='type' href='typenum/operator_aliases/type.Sum.html' title='typenum::operator_aliases::Sum'>Sum</a>&lt;U, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt;: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a></span>","impl <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>&gt; for <a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, B: <a class='trait' href='typenum/marker_traits/trait.Bit.html' title='typenum::marker_traits::Bit'>Bit</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;U, B&gt;&gt; for <a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, B: <a class='trait' href='typenum/marker_traits/trait.Bit.html' title='typenum::marker_traits::Bit'>Bit</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/uint/enum.UTerm.html' title='typenum::uint::UTerm'>UTerm</a>&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;U, B&gt;","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ur, <a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt;&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ul, <a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;</span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ur, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt;&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ul, <a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;</span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ur, <a class='enum' href='typenum/bit/enum.B0.html' title='typenum::bit::B0'>B0</a>&gt;&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ul, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;</span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ur, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt;&gt; for <a class='struct' href='typenum/uint/struct.UInt.html' title='typenum::uint::UInt'>UInt</a>&lt;Ul, <a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;, <a class='type' href='typenum/operator_aliases/type.Sum.html' title='typenum::operator_aliases::Sum'>Sum</a>&lt;Ul, Ur&gt;: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/bit/enum.B1.html' title='typenum::bit::B1'>B1</a>&gt;</span>","impl&lt;I: <a class='trait' href='typenum/marker_traits/trait.Integer.html' title='typenum::marker_traits::Integer'>Integer</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;I&gt; for <a class='enum' href='typenum/int/enum.Z0.html' title='typenum::int::Z0'>Z0</a>","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/int/enum.Z0.html' title='typenum::int::Z0'>Z0</a>&gt; for <a class='struct' href='typenum/int/struct.PInt.html' title='typenum::int::PInt'>PInt</a>&lt;U&gt;","impl&lt;U: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='enum' href='typenum/int/enum.Z0.html' title='typenum::int::Z0'>Z0</a>&gt; for <a class='struct' href='typenum/int/struct.NInt.html' title='typenum::int::NInt'>NInt</a>&lt;U&gt;","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/int/struct.PInt.html' title='typenum::int::PInt'>PInt</a>&lt;Ur&gt;&gt; for <a class='struct' href='typenum/int/struct.PInt.html' title='typenum::int::PInt'>PInt</a>&lt;Ul&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;, Ul::Output: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a></span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/int/struct.NInt.html' title='typenum::int::NInt'>NInt</a>&lt;Ur&gt;&gt; for <a class='struct' href='typenum/int/struct.NInt.html' title='typenum::int::NInt'>NInt</a>&lt;Ul&gt; <span class='where'>where Ul: <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;Ur&gt;, Ul::Output: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a></span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/int/struct.NInt.html' title='typenum::int::NInt'>NInt</a>&lt;Ur&gt;&gt; for <a class='struct' href='typenum/int/struct.PInt.html' title='typenum::int::PInt'>PInt</a>&lt;Ul&gt; <span class='where'>where Ul: <a class='trait' href='typenum/type_operators/trait.Cmp.html' title='typenum::type_operators::Cmp'>Cmp</a>&lt;Ur&gt; + PrivateIntegerAdd&lt;Ul::Output, Ur&gt;</span>","impl&lt;Ul: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>, Ur: <a class='trait' href='typenum/marker_traits/trait.Unsigned.html' title='typenum::marker_traits::Unsigned'>Unsigned</a> + <a class='trait' href='typenum/marker_traits/trait.NonZero.html' title='typenum::marker_traits::NonZero'>NonZero</a>&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='typenum/int/struct.PInt.html' title='typenum::int::PInt'>PInt</a>&lt;Ur&gt;&gt; for <a class='struct' href='typenum/int/struct.NInt.html' title='typenum::int::NInt'>NInt</a>&lt;Ul&gt; <span class='where'>where Ur: <a class='trait' href='typenum/type_operators/trait.Cmp.html' title='typenum::type_operators::Cmp'>Cmp</a>&lt;Ul&gt; + PrivateIntegerAdd&lt;Ur::Output, Ul&gt;</span>",];implementors['diffgeom'] = ["impl&lt;T, U&gt; <a class='trait' href='https://doc.rust-lang.org/nightly/core/ops/trait.Add.html' title='core::ops::Add'>Add</a>&lt;<a class='struct' href='diffgeom/tensors/struct.Tensor.html' title='diffgeom::tensors::Tensor'>Tensor</a>&lt;T, U&gt;&gt; for <a class='struct' href='diffgeom/tensors/struct.Tensor.html' title='diffgeom::tensors::Tensor'>Tensor</a>&lt;T, U&gt; <span class='where'>where T: <a class='trait' href='diffgeom/coordinates/trait.CoordinateSystem.html' title='diffgeom::coordinates::CoordinateSystem'>CoordinateSystem</a>, U: <a class='trait' href='diffgeom/tensors/trait.Variance.html' title='diffgeom::tensors::Variance'>Variance</a>, T::Dimension: <a class='trait' href='typenum/type_operators/trait.Pow.html' title='typenum::type_operators::Pow'>Pow</a>&lt;U::Rank&gt;, Power&lt;T::Dimension, U::Rank&gt;: <a class='trait' href='generic_array/trait.ArrayLength.html' title='generic_array::ArrayLength'>ArrayLength</a>&lt;<a class='primitive' href='https://doc.rust-lang.org/nightly/std/primitive.f64.html'>f64</a>&gt;</span>",];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()
