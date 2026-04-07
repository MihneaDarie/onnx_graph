#[macro_export]
macro_rules! zeros_from_others_type {
    ($other:expr, $shape:expr, [$($variant:ident),+]) => {
        match $other {
            $(
                TypedArray::$variant(_) => TypedArray::$variant(ArrayD::zeros($shape)),
            )+
            TypedArray::Bool(_) => TypedArray::Bool(ArrayD::from_elem($shape, false)),
            _ => TypedArray::Undefined
                        }
        };
}

#[macro_export]
macro_rules! shape_macro {
    ($array:expr, [$($variant:ident),+]) => {
        match $array {
            $(
                TypedArray::$variant(inner) => Some(inner.shape()),
            )+
            TypedArray::Undefined => {
                None
            }
        }
    };
}

#[macro_export]
macro_rules! len_macro {
    ($array:expr, [$($variant:ident),+]) => {
        match $array {
            $(
                TypedArray::$variant(inner) => Some(inner.len()),
            )+
            TypedArray::Undefined => {
                None
            }
        }
    };
}

#[macro_export]
macro_rules! discriminant_macro {
    ($array:expr, [$($variant:ident),+]) => {
        match $array {
            $(
                TypedArray::$variant(_) => stringify!($variant),
            )+
            TypedArray::Undefined => "Undefinde"
        }.to_string()
    };
}

#[macro_export]
macro_rules! zeros_from_datatype {
    ($data_type:expr, $shape:expr, [$($variant:ident),+]) => {
        match $data_type {
            $(
                DataType::$variant => TypedArray::$variant(ArrayD::zeros($shape)),
            )+
            DataType::Bool => TypedArray::Bool(ArrayD::from_elem($shape, false)),
             _ => TypedArray::Undefined
        }
    };
}

macro_rules! cast_to {
    ($data_type:expr, $src_arr:expr) => {};
}

#[macro_export]
macro_rules! cast_to_dst {
    ($arr_base:expr, $data_type:expr, $out:expr, $T_src:ty, [$(($variant_dst:ident, $T_dst:ty)),+]) => {
        match $data_type {
            $(
                DataType::$variant_dst => {
                        if let TypedArray::$variant_dst(out_array) = $out {
                            let out_slice = out_array.as_slice_memory_order_mut().unwrap();
                            $arr_base.as_slice_memory_order()
                                .unwrap()
                                .par_iter()
                                .zip(out_slice.par_iter_mut())
                                .for_each(|(src, dst)| *dst = *src as $T_dst);
                    }
                }
            )+
            DataType::Bool => {
                    if let TypedArray::Bool(out_array) = $out {
                        let out_slice = out_array.as_slice_memory_order_mut().unwrap();
                        $arr_base.as_slice_memory_order()
                            .unwrap()
                            .par_iter()
                            .zip(out_slice.par_iter_mut())
                            .for_each(|(src, dst)| *dst = *src != (0 as $T_src));
                    }
                }
            _ => anyhow::bail!("Can't cast to unsupported array!"),
        }
    };
}

#[macro_export]
macro_rules! cast_bool_to_dst {
    ($arr_base:expr, $data_type:expr, $out:expr, [$(($variant_dst:ident, $T_dst:ty)),+]) => {
        match $data_type {
            $(
                DataType::$variant_dst => {
                    if let TypedArray::$variant_dst(out_array) = $out {
                        let out_slice = out_array.as_slice_memory_order_mut().unwrap();
                        $arr_base.as_slice_memory_order()
                            .unwrap()
                            .par_iter()
                            .zip(out_slice.par_iter_mut())
                            .for_each(|(src, dst)| *dst = if *src == true {1 as $T_dst} else {0 as $T_dst});
                    }
                }
            )+
            DataType::Bool => {
                    if let TypedArray::Bool(out_array) = $out {
                        let out_slice = out_array.as_slice_memory_order_mut().unwrap();
                        $arr_base.as_slice_memory_order()
                            .unwrap()
                            .par_iter()
                            .zip(out_slice.par_iter_mut())
                            .for_each(|(src, dst)| *dst = *src);
                    }
                }
            _ => anyhow::bail!("Can't cast to unsupported array!"),
        }
    };
}

#[macro_export]
macro_rules! copy_and_cast_from_datatype {
    ($data_type:expr, $src:expr, $out:expr, [$(($variant_src:ident, $T_src:ty)),+], $dst_list:tt) => {
        use ndarray::ArrayD;
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;

        match $src {
            $(
                TypedArray::$variant_src(arr_base) => {
                    $crate::cast_to_dst!(arr_base, $data_type, $out, $T_src, $dst_list)
                }
            )+
            TypedArray::Bool(array_base) => {
                $crate::cast_bool_to_dst!(array_base, $data_type, $out, $dst_list)
            }
            _ => anyhow::bail!("Can't cast unsupported array!"),
        }
    };
}

#[macro_export]
macro_rules! fix_if_not_contignous {
    ($self:expr, [$($variant:ident),+]) => {
        match $self {
        $(TypedArray::$variant(a) =>{ if a.is_standard_layout() {
            TypedArray::$variant(a)
        } else {
            TypedArray::$variant(a.as_standard_layout().into_owned())
        }})+,
        other => other,
    }
    };
}

#[macro_export]
macro_rules! fill_from_elem {
    ($self:expr, $shape:expr, $o:expr, [$(($variant:ident,$T:ty)),+]) => {
        match $self {
            $(
                TypedArray::$variant(v) => {
                    let fill = v.iter().next().copied().unwrap_or(<$T>::default());
                    *($o) = TypedArray::$variant(ArrayD::from_elem(IxDyn(&($shape)), fill));
                }
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for fill")),
        }
    };
}

#[macro_export]
macro_rules! get_curent_size_and_shape {
    ($self:expr, [$($variant:ident),+]) => {
        match $self {
            $(
            TypedArray::$variant(a) => (a.len(), a.shape().to_vec()),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for reshape")),
        }
    };
}

#[macro_export]
macro_rules! from_shape_vec_from_datatype {
    ($data_type:expr, $shape: expr, $data:expr, [$(($variant:ident, $T:ty)),+]) => {
        match $data_type {
            $(
                DataType::$variant => {
                        let vec = $data
                                  .chunks_exact(std::mem::size_of::<$T>())
                                  .map(|b| <$T>::from_le_bytes(b.try_into().unwrap()))
                                  .collect::<Vec<$T>>();
                    TypedArray::$variant(ArrayD::from_shape_vec($shape, vec).unwrap())
                }
            )+
            DataType::Bool => {
                let bools = $data.iter().map(|&b| b != 0).collect::<Vec<bool>>();
                    TypedArray::Bool(ArrayD::from_shape_vec($shape, bools).unwrap())
            }
            _ => TypedArray::Undefined,
        }
    };
}
