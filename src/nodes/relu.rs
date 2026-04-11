use std::any::Any;

use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::apply_relu;

use crate::{
    call_activation_source_to_destination,
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Default)]
pub struct ReluNode<T: Default> {
    pub x: String,

    pub o: String,

    unique_id: UniqueId,

    pub next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> ReluNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut relu = Self {
            x: String::new(),
            o: String::new(),
            unique_id: UniqueId::Relu,
            next_node: None,
        };
        relu.add_input_strings(elem.inputs[0].clone());
        relu.add_output_strings(elem.outputs[0].clone());
        relu
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ReluNode<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }
    fn get_unique_id_mut(&mut self) -> UniqueId {
        self.unique_id
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

    fn input_names(&self) -> Vec<String> {
        vec![self.x.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("relu-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                x.relu(result).unwrap();
            }
            None => panic!("ReluNode: missing input {}", self.x),
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

#[inline(always)]
pub fn relu_f64(x: f64) -> f64 {
    x.max(0.0f64)
}

#[inline(always)]
pub fn relu_f32(x: f32) -> f32 {
    x.max(0.0f32)
}

impl TypedArray {
    call_activation_source_to_destination!(
        relu,
        Some(apply_relu),
        [(Float, relu_f32), (Double, relu_f64)]
    );
}
