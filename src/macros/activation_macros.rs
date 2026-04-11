#[macro_export]
macro_rules! call_activation_source_to_destination {
    ($func_name:ident, $f32_simd_option:expr, [$(($variant:ident,$specific_func:ident)),+]) => {
        pub fn $func_name(&self, o: &mut TypedArray) -> anyhow::Result<()> {
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::IntoParallelRefMutIterator;
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::ParallelIterator;

            let in_shape = self.shape().ok_or_else(|| anyhow::anyhow!("undefined input"))?;

            match self {
                $(
                    TypedArray::$variant(_) => {
                        let needs_alloc = match &*o {
                            TypedArray::$variant(out) => out.shape() != in_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::empty_with_others_type(self, in_shape);
                        }
                    }
                )+
                _ => return Err(anyhow::anyhow!("{} only supported for given types", stringify!($func_name))),
            }

            if let (TypedArray::Float(i), TypedArray::Float(o), Some(func)) = (self, &mut *o, $f32_simd_option) {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                func(dst, src);
                return Ok(());
            }

            match (self, &mut *o) {
                $(
                    (TypedArray::$variant(a), TypedArray::$variant(o)) => {
                        let src = a.as_slice_memory_order().unwrap();
                        let dst = o.as_slice_memory_order_mut().unwrap();
                        dst.par_iter_mut()
                            .zip(src.par_iter())
                            .for_each(|(d, s)| *d = $specific_func(*s));
                    }
                )+
                _ => return Err(anyhow::anyhow!("{} only supported for given types", stringify!($func_name))),
            };
            Ok(())
        }
    };
}
