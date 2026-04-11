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
                                *o = TypedArray::$variant(a $op b).ensure_contiguous();
                            };

                        } else {
                            *o = TypedArray::$variant(a $op b).ensure_contiguous();
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
            use ndarray::IxDyn;
            use ndarray::ArrayD;

            match (self, b) {
                $(
                    (TypedArray::$variant(a), TypedArray::$variant(b)) => {
                        let out_shape = crate::nodes::where_op::WhereNode::<f32>::broadcast_shape(
                            &[a.shape(), b.shape()]
                        )?;

                        let needs_alloc = match &*o {
                            TypedArray::Bool(out) => out.shape() != out_shape.as_slice(),
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Bool(ArrayD::from_elem(IxDyn(&out_shape), false));
                        }

                        if let TypedArray::Bool(out) = o {
                            if a.shape() == b.shape() {
                                let dst = out.as_slice_memory_order_mut().unwrap();
                                let sa = a.as_slice_memory_order().unwrap();
                                let sb = b.as_slice_memory_order().unwrap();
                                dst.iter_mut()
                                    .zip(sa.iter().zip(sb.iter()))
                                    .for_each(|(d, (a, b))| *d = $op(a, b));
                            } else {
                                let a_bc = a.broadcast(IxDyn(&out_shape)).unwrap();
                                let b_bc = b.broadcast(IxDyn(&out_shape)).unwrap();
                                ndarray::Zip::from(out)
                                    .and(&a_bc)
                                    .and(&b_bc)
                                    .for_each(|o, a, b| *o = $op(a, b));
                            }
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
