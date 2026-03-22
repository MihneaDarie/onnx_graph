#[macro_export]
macro_rules! call_transpose_for_typed_array {
    ($self:expr, $perm:expr, $o:expr, [$($variant:ident),+]) => {
        match $self {
            $(
                TypedArray::$variant(a) => transpose_variant!($variant, $perm, a, $o),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for transpose")),
        }
    };
}

#[macro_export]
macro_rules! transpose_variant {
    ($variant:ident, $perm:expr, $a:expr, $o:expr) => {{
        let ndim = $a.ndim();
        let perm: Vec<usize> = if $perm.is_empty() {
            (0..ndim).rev().collect()
        } else {
            $perm
                .iter()
                .map(|&p| {
                    if p < 0 {
                        (ndim as i64 + p) as usize
                    } else {
                        p as usize
                    }
                })
                .collect()
        };

        let view = $a.view().permuted_axes(&*perm);

        if let TypedArray::$variant(out) = $o {
            ndarray::Zip::from(out).and(&view).par_for_each(|d, s| {
                *d = *s;
            });
        }
    }};
}
