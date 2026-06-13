use std::{
    any::Any,
    hash::{DefaultHasher, Hash, Hasher},
};

use ndarray::{ArrayD, IxDyn};
use onnx_extractor::OnnxOperation;

use crate::{
    impl_typed_binop,
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::hash_string,
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Default)]
pub struct AndNode<T: Default> {
    a: u64,
    b: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> AndNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut and = Self {
            a: u64::default(),
            b: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::And,
            next_node: None,
        };
        let aid = hash_string(&elem.inputs()[0]);
        let bid = hash_string(&elem.inputs()[1]);
        and.add_inputs(aid, bid);

        let oid = hash_string(&elem.outputs()[0]);
        and.add_outputs(oid);
        and
    }
    pub fn add_inputs(&mut self, a: u64, b: u64) {
        self.a = a;
        self.b = b;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for AndNode<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }
    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>> {
        self.next_node.take()
    }
    fn get_unique_id_mut(&mut self) -> UniqueId {
        self.unique_id
    }

    fn get_next_mut(&mut self) -> Option<&mut Vec<Box<dyn Node<T>>>> {
        self.next_node.as_mut()
    }

    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>) {
        self.next_node = next;
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(out) => {
                a.and_op(b, out).unwrap();
            }
            _ => panic!("AndNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("and-{},{},{}", self.a, self.b, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [a, o] = omap.get_disjoint_mut([&self.a, &self.o]);
        let a = a.map(|arr| &*arr);

        if let (Some(a), Some(o)) = (a, o)
            && let Some(in_shape) = a.shape()
        {
            *o = TypedArray::Bool(ArrayD::default(IxDyn(in_shape))).ensure_contiguous();
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    impl_typed_binop!(and_op, &, [Bool]);
}
