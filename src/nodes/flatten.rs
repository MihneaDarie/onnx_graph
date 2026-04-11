use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::{Ok, Result};
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct FlattenNode<T: Default> {
    x: String,

    axis: Option<i64>,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for FlattenNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let mut flatten = Self {
            x: String::new(),
            axis: Some(0),
            o: String::new(),
            unique_id: UniqueId::Flatten,
            next_node: None,
        };
        let attrs = &elem.attributes;
        let axis = attrs.get("axis").and_then(|val| val.as_int());
        flatten.axis = axis;
        flatten.add_input_strings(elem.inputs[0].clone());
        flatten.add_output_strings(elem.outputs[0].clone());
        Ok(flatten)
    }
}

impl<T: Default> FlattenNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut flatten = Self {
            x: String::new(),
            o: String::new(),
            unique_id: UniqueId::Flatten,
            next_node: None,
            axis: Some(0),
        };
        flatten.add_input_strings(elem.inputs[0].clone());
        flatten.add_output_strings(elem.outputs[0].clone());
        flatten
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for FlattenNode<T> {
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

        match o {
            Some(result) => {
                let axis = self.axis.unwrap();
                x.flatten_op(axis, result).unwrap();
            }
            None => panic!("FlattenNode: missing input {}", self.x),
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
        println!("flatten-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let (Some(in_shape), Some(axis)) = (x.shape(), self.axis)
        {
            let ndim = in_shape.len();
            let axis = if axis < 0 {
                (ndim as i64 + axis) as usize
            } else {
                axis as usize
            };

            let dim0: usize = in_shape[..axis].iter().product();
            let dim1: usize = in_shape[axis..].iter().product();

            *o = TypedArray::empty_with_others_type(x, &[dim0, dim1]);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    pub fn flatten_op(&self, axis: i64, o: &mut TypedArray) -> anyhow::Result<()> {
        use ndarray::ArrayD;
        use ndarray::IxDyn;

        let rank = self.shape().unwrap().len() as i64;
        let axis = if axis < 0 { axis + rank } else { axis } as usize;

        let shape = self.shape().unwrap();
        let dim0: usize = shape[..axis].iter().product::<usize>().max(1);
        let dim1: usize = shape[axis..].iter().product::<usize>().max(1);
        let out_shape = [dim0, dim1];

        macro_rules! flatten_variant {
            ($variant:ident, $a:expr) => {{
                let needs_alloc = match &*o {
                    TypedArray::$variant(out) => out.shape() != out_shape,
                    _ => true,
                };
                if needs_alloc {
                    *o = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape)));
                }
                if let TypedArray::$variant(out) = o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    let src = $a.as_slice_memory_order().unwrap();
                    dst.copy_from_slice(src);
                }
            }};
        }

        macro_rules! call_flatten_for_typed_array {
            ([$($variant:ident),+]) => {

                match self {
                    $(
                        TypedArray::$variant(a) => flatten_variant!($variant, a),
                    )+
                    TypedArray::Bool(a) => {
                        let needs_alloc = match &*o {
                            TypedArray::Bool(out) => out.shape() != out_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Bool(ArrayD::from_elem(IxDyn(&out_shape), false));
                        }
                        if let TypedArray::Bool(out) = o {
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            let src = a.as_slice_memory_order().unwrap();
                            dst.copy_from_slice(src);
                        }
                    }
                    _ => anyhow::bail!("Flatten: unsupported type"),
                }
            };
        }

        call_flatten_for_typed_array!([
            Double, Float, Int16, Int32, Int64, Int8, Uint16, Uint32, Uint64, Uint8
        ]);

        Ok(())
    }
}
