#[macro_export]
macro_rules! call_gather_for_typed_array {
    ($data:expr, $axis:expr, $idx_vec:expr, $idx_shape:expr, $o:expr, [$($variant:ident),+]) => {
        use crate::gather_variant;

        match $data {
            $(
                TypedArray::$variant(arr) => gather_variant!($variant, $axis, $idx_vec, $idx_shape, arr, $o),
            )+
            _ => return Err(anyhow::anyhow!("argmax: unsupported type")),
        }
    };
}

#[macro_export]
macro_rules! gather_variant {
    ($variant:ident, $axis:expr, $idx_vec:expr, $idx_shape:expr, $arr:expr, $o:expr) => {{
        let ndim = $arr.ndim() as i64;
        let axis_usize = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let data_shape = $arr.shape();
        let axis_size = data_shape[axis_usize] as i64;

        let mut out_shape: Vec<usize> = Vec::new();
        for i in 0..axis_usize {
            out_shape.push(data_shape[i]);
        }

        for &s in &($idx_shape) {
            out_shape.push(s);
        }
        for i in (axis_usize + 1)..data_shape.len() {
            out_shape.push(data_shape[i]);
        }

        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let needs_alloc = match &*($o) {
            TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
            _ => true,
        };

        if needs_alloc {
            *($o) = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape)));
        }

        let out_arr = match $o {
            TypedArray::$variant(arr) => arr,
            _ => unreachable!(),
        };

        let data_sl = $arr.as_slice_memory_order().unwrap();
        let out_sl = out_arr.as_slice_memory_order_mut().unwrap();

        let outer_size: usize = data_shape[..axis_usize].iter().product();
        let inner_size: usize = data_shape[axis_usize + 1..].iter().product();
        let axis_dim = data_shape[axis_usize];

        let mut out_idx = 0;
        for outer in 0..outer_size {
            for &idx in &($idx_vec) {
                let idx = if idx < 0 {
                    (axis_size + idx) as usize
                } else {
                    idx as usize
                };

                let src_offset = outer * axis_dim * inner_size + idx * inner_size;
                let len = inner_size;

                out_sl[out_idx..out_idx + len]
                    .copy_from_slice(&data_sl[src_offset..src_offset + len]);
                out_idx += len;
            }
        }
    }};
}
