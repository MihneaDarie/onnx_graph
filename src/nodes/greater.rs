use std::{any::Any, collections::HashMap};

use crate::{
    impl_typed_binop_with_boolean_output,
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct GreaterNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> GreaterNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut greater = Self {
            a: String::new(),
            b: String::new(),
            o: String::new(),
            unique_id: UniqueId::Greater,
            next_node: None,
        };
        greater.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        greater.add_output_strings(elem.outputs[0].clone());
        greater
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for GreaterNode<T> {
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
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(result) => {
                a.greater_op(b, result).unwrap();
            }
            None => panic!("GreaterNode: missing input {}", self.a),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.a.clone()]
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
        println!("greater-{},{}", self.a, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.a, &self.o]);
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

impl TypedArray {
    impl_typed_binop_with_boolean_output!(
        greater_op,
        |a, b| a > b,
        [
            Double, Float, Int16, Int32, Int64, Int8, Uint16, Uint32, Uint64, Uint8
        ]
    );
}
