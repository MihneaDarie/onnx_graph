#[macro_export]
macro_rules! softmax_variant {
    ($variant:ident, $axis:expr, $a:expr, $o:expr, $T:ty) => {{
        let ndim = $a.ndim() as i64;
        let axis = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        if let TypedArray::$variant(out) = $o {
            let dst = out.as_slice_memory_order_mut().unwrap();
            let src = $a.as_slice_memory_order().unwrap();
            dst.copy_from_slice(src);

            for mut lane in out.lanes_mut(Axis(axis)) {
                let max = lane.iter().copied().fold(<$T>::NEG_INFINITY, <$T>::max);
                lane.mapv_inplace(|x| (x - max).exp());
                let sum: $T = lane.iter().sum();
                lane.mapv_inplace(|x| x / sum);
            }
        }
    }};
}
