#[macro_export]
macro_rules! call_split_for_typed_array {
    ($self:expr, $axis:expr, $splits:expr, $outputs:expr, [$($variant:ident),+]) => {
        match $self {
            $(
                TypedArray::$variant(a) => split_variant!($variant, $axis, $splits, a, $outputs),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for split")),
        }
    };
}

#[macro_export]
macro_rules! split_variant {
    ($variant:ident, $axis:expr, $splits:expr, $a:expr, $outputs:expr) => {{
        let ndim = $a.ndim() as i64;
        let axis = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let mut offset = 0;
        for &size in $splits.iter() {
            let size = size as usize;
            let slice_info: Vec<ndarray::SliceInfoElem> = (0..$a.ndim())
                .map(|i| {
                    if i == axis {
                        ndarray::SliceInfoElem::Slice {
                            start: offset as isize,
                            end: Some((offset + size) as isize),
                            step: 1,
                        }
                    } else {
                        ndarray::SliceInfoElem::Slice {
                            start: 0,
                            end: None,
                            step: 1,
                        }
                    }
                })
                .collect();

            $outputs.push(TypedArray::$variant(
                $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?)
                    .to_owned(),
            ));
            offset += size;
        }
    }};
}
