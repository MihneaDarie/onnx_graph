#[macro_export]
macro_rules! impl_typed_singleopfunction_with_the_same_output_type_as_the_output {
    ($name:ident, $method:ident, [$($variant:ident),+], [$($reject:ident),*]) => {
        pub fn $name(&self, o: &mut TypedArray) -> anyhow::Result<()> {
            match self {
                $(
                    TypedArray::$variant(a) => {
                        let needs_alloc = match &*o {
                            TypedArray::$variant(out) => out.shape() != a.shape(),
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::$variant(ArrayD::zeros(IxDyn(a.shape())));
                        }
                        if let TypedArray::$variant(out) = o {
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            let src = a.as_slice_memory_order().unwrap();
                            dst.par_iter_mut()
                                .zip(src.par_iter())
                                .for_each(|(d, s)| *d = (*s).$method());
                        }
                    }
                )+
                $(
                    TypedArray::$reject(_) => {
                        anyhow::bail!("unsupported type: {}", stringify!($reject));
                    }
                )*
                TypedArray::Undefined => {
                    anyhow::bail!("undefined type");
                }
                _ => anyhow::bail!("unsupported type"),
            }
            Ok(())
        }
    };
}


#[macro_export]
macro_rules! impl_typed_singleopfunction_with_boolean_ouput {
    ($name:ident, $method:ident, [$($variant:ident),+], [$($reject:ident),*]) => {
        pub fn $name(&self, o: &mut TypedArray) -> anyhow::Result<()> {
            match self {
                $(
                    TypedArray::$variant(a) => {
                        let needs_alloc = match &*o {
                            TypedArray::Bool(out) => out.shape() != a.shape(),
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Bool(ArrayD::from_elem(a.shape(), false));
                        }
                        if let TypedArray::Bool(out) = o {
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            let src = a.as_slice_memory_order().unwrap();
                            dst.par_iter_mut()
                                .zip(src.par_iter())
                                .for_each(|(d, s)| *d = (*s).$method());
                        }
                    }
                )+
                $(
                    TypedArray::$reject(_) => {
                        anyhow::bail!("unsupported type: {}", stringify!($reject));
                    }
                )*
                TypedArray::Undefined => {
                    anyhow::bail!("undefined type");
                }
                _ => anyhow::bail!("unsupported type"),
            }
            Ok(())
        }
    };
}
