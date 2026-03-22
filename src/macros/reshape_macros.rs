#[macro_export]
macro_rules! call_reshape_for_typed_array {
    ($self:expr, $new_shape:expr, $o:expr, [$($variant:ident),+]) => {
        match $self {
            $(
                TypedArray::$variant(a) => reshape_variant!($variant, $new_shape, a, $o),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for reshape")),
        }
    };
}

#[macro_export]
macro_rules! reshape_variant {
    ($variant:ident, $new_shape:expr ,$a:expr, $o:expr) => {{
        let src = $a.as_slice_memory_order().unwrap();

        let needs_realloc = match &*($o) {
            TypedArray::$variant(out) => out.shape() != $new_shape.as_slice(),
            _ => true,
        };

        if needs_realloc {
            *($o) =
                TypedArray::$variant(ArrayD::from_shape_vec(IxDyn(&($new_shape)), src.to_vec())?);
        } else {
            if let TypedArray::$variant(out) = $o {
                let dst = out.as_slice_memory_order_mut().unwrap();
                dst.copy_from_slice(src);
            }
        }
    }};
}
