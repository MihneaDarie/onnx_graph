#[macro_export]
macro_rules! impl_typed_binop {
    ($name:ident, $op:tt, [$($variant:ident),+]) => {
        pub fn $name(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::IntoParallelRefMutIterator;
            use rayon::iter::ParallelIterator;

            match (self, b) {
                $(
                    (TypedArray::$variant(a), TypedArray::$variant(b)) => {
                        if a.shape() == b.shape() {
                            if let TypedArray::$variant(out) = o {
                                    let dst = out.as_slice_memory_order_mut().unwrap();
                                    let sa = a.as_slice_memory_order().unwrap();
                                    let sb = b.as_slice_memory_order().unwrap();
                                    dst.par_iter_mut()
                                        .zip(sa.par_iter().zip(sb.par_iter()))
                                        .for_each(|(d, (a, b))| *d = *a $op *b);
                            } else {
                                *o = TypedArray::$variant(a $op b);
                            };

                        } else {
                            *o = TypedArray::$variant(a $op b);
                        }
                    }
                )+
                (TypedArray::Undefined, _) | (_, TypedArray::Undefined) => {
                    return Err(anyhow::anyhow!("undefined type: {}", stringify!($name)));
                }
                _ => return Err(anyhow::anyhow!("mismatch types {}", stringify!($name))),
            }
            Ok(())
        }
    };
}

#[macro_export]
macro_rules! impl_typed_binop_with_boolean_output {
    ($name:ident, $op:expr, [$($variant:ident),+]) => {
        pub fn $name(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::IntoParallelRefMutIterator;
            use rayon::iter::ParallelIterator;

            match (self, b) {
                $(
                    (TypedArray::$variant(a), TypedArray::$variant(b)) => {
                        if a.shape() == b.shape() {
                            if let TypedArray::Bool(out) = o {
                                let dst = out.as_slice_memory_order_mut().unwrap();
                                let sa = a.as_slice_memory_order().unwrap();
                                let sb = b.as_slice_memory_order().unwrap();
                                dst.par_iter_mut()
                                    .zip(sa.par_iter().zip(sb.par_iter()))
                                    .for_each(|(d, (a, b))| *d = $op(a, b));
                            } else {
                                *o = TypedArray::Bool(
                                    ndarray::Zip::from(a)
                                        .and(b)
                                        .map_collect(|a, b| $op(a, b))
                                );
                            }
                        } else {
                            return Err(anyhow::anyhow!(
                                "shape mismatch in {}", stringify!($name)
                            ));
                        }
                    }
                )+
                (TypedArray::Undefined, _) | (_, TypedArray::Undefined) => {
                    return Err(anyhow::anyhow!("undefined type: {}", stringify!($name)));
                }
                _ => return Err(anyhow::anyhow!("mismatch types {}", stringify!($name))),
            }
            Ok(())
        }
    };
}
