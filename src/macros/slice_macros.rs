#[macro_export]
macro_rules! call_slice_for_typed_array {
    ($self:expr, $axes:expr, $starts:expr, $ends:expr, $o:expr, [$($variant:ident),+]) => {
        match $self {
            $(
                TypedArray::$variant(a) => slice_variant!($variant, $axes, $starts, $ends, a, $o),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for slice")),
        }
    };
}

#[macro_export]
macro_rules! slice_variant {
    ($variant:ident, $axes:expr, $starts:expr, $ends:expr, $a:expr, $o:expr) => {{
        let ndim = $a.ndim();
        let mut slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
            .map(|_| ndarray::SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();

        for i in 0..$axes.len() {
            let axis = $axes[i] as usize;
            let dim_size = $a.shape()[axis] as i64;

            let start = {
                let s = $starts[i];
                if s < 0 {
                    (dim_size + s).max(0)
                } else {
                    s.min(dim_size)
                }
            } as usize;

            let end = {
                let e = $ends[i];
                if e < 0 {
                    (dim_size + e).max(0)
                } else {
                    e.min(dim_size)
                }
            } as usize;

            slice_info[axis] = ndarray::SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            };
        }

        let view = $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?);

        if let TypedArray::$variant(out) = $o {
            let dst = out.as_slice_memory_order_mut().unwrap();
            for (d, s) in dst.iter_mut().zip(view.iter()) {
                *d = *s;
            }
        }
    }};
}
