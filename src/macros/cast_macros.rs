macro_rules! cast_to {
    ($arr:expr, $o:expr, $to:expr, ) => {
        match $o {
            TypedArray::Int8(a) => {}
        }
    };
}

#[macro_export]
macro_rules! impl_cast_for_typedarray {
    ([$(($variant:ident,$T:ty)),+]) => {
        pub fn cast(&self,  o: &mut TypedArray, to: DataType) -> anyhow::Result<()> {
            let in_shape = self.shape().unwrap();
            match self {
                match to {
                    $(
                        DataType::$variant => impl_cast_for_variant(a, o, in_shape, $variant, $T),
                    )+
                    DataType::Bool => {
                        let needs_alloc = match &o {
                            TypedArray::Bool(out) => out.shape() != in_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Bool(ArrayD::from_elem(in_shape,false));
                        }

                        if let TypedArray::Bool(out) = o {
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            let src = x_arr.as_slice_memory_order().unwrap();

                            dst.par_iter_mut()
                            .zip(src.par_iter())
                            .for_each(|(d,s)| *d = *s as bool)
                        }
                        Ok(())
                    }
                    _ => anyhow::bail!("Invalid cast !")
            };
            }
                Ok(())

        }
    };
}

#[macro_export]
macro_rules! impl_cast_for_variant {
    ($x_arr:expr, $o:expr, $in_shape:expr, $variant:ident, $T:ty) => {{
        let needs_alloc = match &($o) {
            TypedArray::$variant(out) => out.shape() != $in_shape,
            _ => true,
        };
        if needs_alloc {
            *($o) = TypedArray::$variant(ArrayD::zeros(in_shape));
        }

        if let TypedArray::$variant(out) = $o {
            let dst = out.as_slice_memory_order_mut().unwrap();
            let src = $x_arr.as_slice_memory_order().unwrap();

            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = *s as $T)
        }
        Ok(())
    }};
}
