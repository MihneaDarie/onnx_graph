#[macro_export]
macro_rules! call_argmax_for_typed_array {
    ($data:expr, $axis:expr, $keepdims:expr, $select_last_index:expr, $o:expr, [$($variant:ident),+]) => {
        match $data {
            $(
                TypedArray::$variant(arr) => argmax_variant!(arr, $axis, $keepdims, $select_last_index, $o),
            )+
            _ => return Err(anyhow::anyhow!("argmax: unsupported type")),
        }
    };
}

#[macro_export]
macro_rules! argmax_variant {
    ($arr:expr, $axis:expr, $keepdims:expr, $select_last_index:expr, $o:expr) => {{
        let ndim = $arr.ndim() as i64;
        let axis_usize = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let mut out_shape: Vec<usize> = $arr.shape().to_vec();
        let axis_len = out_shape[axis_usize];

        if $keepdims {
            out_shape[axis_usize] = 1;
        } else {
            out_shape.remove(axis_usize);
        }

        let needs_alloc = match &*($o) {
            TypedArray::Int64(out) => out.shape() != out_shape.as_slice(),
            _ => true,
        };

        if needs_alloc {
            *($o) = TypedArray::Int64(ArrayD::zeros(IxDyn(&out_shape)));
        }

        let out_arr = match $o {
            TypedArray::Int64(arr) => arr,
            _ => unreachable!(),
        };

        let out_sl = out_arr.as_slice_memory_order_mut().unwrap();
        let mut idx = 0;

        for lane in $arr.lanes(Axis(axis_usize)) {
            let mut max_val = lane[0];
            let mut max_idx: i64 = 0;

            for i in 1..axis_len {
                let val = lane[i];
                if $select_last_index {
                    if val >= max_val {
                        max_val = val;
                        max_idx = i as i64;
                    }
                } else {
                    if val > max_val {
                        max_val = val;
                        max_idx = i as i64;
                    }
                }
            }

            out_sl[idx] = max_idx;
            idx += 1;
        }
    }};
}
