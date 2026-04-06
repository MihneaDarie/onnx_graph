#[macro_export]
macro_rules! call_pow_for_typed_array {
    ($self:expr, $b:expr, $o:expr, $in_shape:expr, [$(($variant:ident, $T:ty)),+]) => {
        use crate::impl_pow_variant;
        use ndarray::ArrayD;
        use ndarray::IxDyn;
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;

        match $self {
            $(
                TypedArray::$variant(a) => impl_pow_variant!($variant, $T, a, $b, $o, $in_shape),
            )+
            _ => anyhow::bail!("Pow: unsupported type for A"),
        }
    };
}

#[macro_export]
macro_rules! impl_pow_variant {
    ($variant: ident, $T:ty, $a_arr:expr, $b:expr, $o:expr, $in_shape:expr) => {{
        let needs_alloc = match &($o) {
            TypedArray::$variant(out) => out.shape() != $in_shape,
            _ => true,
        };
        if needs_alloc {
            *($o) = TypedArray::$variant(ArrayD::zeros(IxDyn($in_shape)));
        }

        if let TypedArray::$variant(out) = $o {
            let dst = out.as_slice_memory_order_mut().unwrap();
            let src = $a_arr.as_slice_memory_order().unwrap();

            macro_rules! pow_for_specific_type {
                (float, $b_array:expr) => {{
                    let b = $b_array
                        .as_slice_memory_order()
                        .expect("Couldn't extract the power of the exponent for floating point !");

                    dst.par_iter_mut()
                        .zip(src.par_iter().zip(b.par_iter()))
                        .for_each(|(d, (s, p))| *d = (*s as f64).powf(*p as f64) as $T);
                }};
                (int, $b_arr:expr) => {{
                    let b = $b_arr
                        .as_slice_memory_order()
                        .expect("Couldn't extract the power of the exponent for integer !");

                    dst.par_iter_mut()
                        .zip(src.par_iter().zip(b.par_iter()))
                        .for_each(|(d, (s, p))| *d = (*s as f64).powi(*p as i32) as $T);
                }};
            }

            match $b {
                TypedArray::Double(b) => pow_for_specific_type!(float, b),
                TypedArray::Float(b) => pow_for_specific_type!(float, b),
                TypedArray::Int64(b) => pow_for_specific_type!(int, b),
                TypedArray::Int32(b) => pow_for_specific_type!(int, b),
                TypedArray::Int16(b) => pow_for_specific_type!(int, b),
                TypedArray::Int8(b) => pow_for_specific_type!(int, b),
                TypedArray::Uint64(b) => pow_for_specific_type!(int, b),
                TypedArray::Uint32(b) => pow_for_specific_type!(int, b),
                TypedArray::Uint16(b) => pow_for_specific_type!(int, b),
                TypedArray::Uint8(b) => pow_for_specific_type!(int, b),
                _ => anyhow::bail!("Pow: unsupported exponent type"),
            }
        }
    }};
}
