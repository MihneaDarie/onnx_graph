use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
    zeros_from_datatype,
};
use anyhow::Result;
use onnx_extractor::{DataType, OnnxOperation};
use std::any::Any;

#[derive(Default)]
pub struct CastNode<T: Default> {
    x: String,

    to: Option<DataType>,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for CastNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let to = attrs
            .get("to")
            .and_then(|v| v.as_int().map(|val| DataType::from_onnx_type(val as i32)));
        let mut cast = Self {
            x: String::new(),
            to,
            o: String::new(),
            unique_id: UniqueId::Cast,
            next_node: None,
        };
        cast.add_input_strings(elem.inputs[0].clone());
        cast.add_output_strings(elem.outputs[0].clone());
        Ok(cast)
    }
}

impl<T: Default> CastNode<T> {
    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for CastNode<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }
    fn get_unique_id_mut(&mut self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = &*x.unwrap();

        match (o, self.to) {
            (Some(result), Some(to)) => {
                x.cast(result, to).unwrap();
            }
            _ => panic!("CastNode: missing input {}", self.x),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.x.clone()]
    }

    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>> {
        self.next_node.take()
    }
    fn get_next_mut(&mut self) -> Option<&mut Vec<Box<dyn Node<T>>>> {
        self.next_node.as_mut()
    }

    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>) {
        self.next_node = next;
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("cast-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
            *o = TypedArray::empty_with_others_type(x, in_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

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
                    cast_to_dst!(arr_base, $data_type, $out, $T_src, $dst_list)
                }
            )+
            TypedArray::Bool(array_base) => {
                cast_bool_to_dst!(array_base, $data_type, $out, $dst_list)
            }
            _ => anyhow::bail!("Can't cast unsupported array!"),
        }
    };
}

impl TypedArray {
    pub fn cast(&self, o: &mut TypedArray, to: DataType) -> anyhow::Result<()> {
        let need_alloc = match (self.shape(), o.shape()) {
            (None, _) => panic!("Undefined input array!"),
            (Some(_), None) => true,
            (Some(in_shape), Some(out_shape)) => in_shape != out_shape || !o.matches_datatype(to),
        };

        if need_alloc {
            let in_shape = self.shape().unwrap();
            *o = zeros_from_datatype!(
                to,
                in_shape,
                [
                    Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
                ]
            );
        }

        copy_and_cast_from_datatype!(
            to,
            self,
            o,
            [
                (Float, f32),
                (Double, f64),
                (Int8, i8),
                (Int16, i16),
                (Int32, i32),
                (Int64, i64),
                (Uint8, u8),
                (Uint16, u16),
                (Uint32, u32),
                (Uint64, u64)
            ],
            [
                (Float, f32),
                (Double, f64),
                (Int8, i8),
                (Int16, i16),
                (Int32, i32),
                (Int64, i64),
                (Uint8, u8),
                (Uint16, u16),
                (Uint32, u32),
                (Uint64, u64)
            ]
        );

        Ok(())
    }
}
