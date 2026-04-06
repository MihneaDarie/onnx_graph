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

    fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.next_node {
            let mut ct = 0;
            let mut sum = 0;
            next.iter().for_each(|val| {
                sum += val.self_count(ct);
                ct += 1;
            });
            sum
        } else {
            count
        }
    }

    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()> {
        if let Some(next_node) = &mut self.next_node {
            next_node[0].insert(next)?;
            return Ok(());
        } else {
            self.next_node = Some(vec![next])
        }
        Ok(())
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
        use ndarray::Dimension;
        use ndarray::IxDyn;

        let rank = self.shape().unwrap().len() as i64;
        let axis = if axis < 0 { axis + rank } else { axis } as usize;

        let shape = self.shape().unwrap();
        let dim0: usize = shape[..axis].iter().product::<usize>().max(1);
        let dim1: usize = shape[axis..].iter().product::<usize>().max(1);
        let out_shape = IxDyn(&[dim0, dim1]);
        macro_rules! flatten_typed {
        ($(($variant:ident, $T:ty)),+) => {
            match (self, &mut *o) {
                $(
                    (TypedArray::$variant(in_arr), TypedArray::$variant(out_arr)) => {
                        if out_arr.len() != dim0 * dim1 {
                            *out_arr = ArrayD::<$T>::zeros(out_shape);
                        } else if out_arr.shape() != out_shape.as_array_view().as_slice().unwrap() {
                            *out_arr = out_arr.clone().into_shape_with_order(out_shape).unwrap();
                        }

                        let out_slice = out_arr.as_slice_memory_order_mut().unwrap();
                        let in_slice = in_arr.as_slice_memory_order().unwrap();

                        out_slice.copy_from_slice(in_slice);

                        Ok(())
                    }
                )+
                (TypedArray::Bool(in_arr), TypedArray::Bool(out_arr)) => {
                    if out_arr.len() != dim0 * dim1 {
                        *out_arr = ArrayD::<bool>::from_elem(out_shape, false);
                    } else if out_arr.shape() != out_shape.as_array_view().as_slice().unwrap() {
                        *out_arr = out_arr.clone().into_shape_with_order(out_shape).unwrap();
                    }

                    let out_slice = out_arr.as_slice_memory_order_mut().unwrap();
                    let in_slice = in_arr.as_slice_memory_order().unwrap();

                    out_slice.copy_from_slice(in_slice);

                    Ok(())
                }
                _ => anyhow::bail!("Flatten: input and output must have the same type"),
            }
        };
    }

        flatten_typed!(
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
        )
    }
}
