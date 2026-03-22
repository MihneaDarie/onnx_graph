#[macro_export]
macro_rules! call_concat_for_typed_array {
    ($first:expr, $arrays:expr, $axis:expr, $o:expr, [$($variant:ident),+]) => {
        match $first {
            $(
                TypedArray::$variant(_) => concat_variant!($variant, $arrays, $axis, $o),
            )+
            TypedArray::Undefined => return Err(anyhow::anyhow!("undefined type in concat")),
            _ => return Err(anyhow::anyhow!("unsupported type for concat")),
        }
    };
}

#[macro_export]
macro_rules! concat_variant {
    ($variant:ident, $arrays:expr, $axis:expr, $o:expr) => {{
        let inner: anyhow::Result<Vec<_>> = $arrays
            .iter()
            .map(|a| match a {
                TypedArray::$variant(arr) => Ok(arr.view()),
                _ => Err(anyhow::anyhow!("type mismatch in concat")),
            })
            .collect();
        *$o = TypedArray::$variant(ndarray::concatenate(Axis($axis), &inner?)?.into_dyn());
    }};
}
