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
