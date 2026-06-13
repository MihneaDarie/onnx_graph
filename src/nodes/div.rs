use std::any::Any;

use crate::{
    impl_typed_binop,
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::hash_string,
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::div_maybe_simd;

#[derive(Default)]
pub struct DivNode<T: Default> {
    a: u64,
    b: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> DivNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut div = Self {
            a: u64::default(),
            b: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Div,
            next_node: None,
        };
        let aid = hash_string(&elem.inputs()[0]);
        let bid = hash_string(&elem.inputs()[1]);
        div.add_inputs(aid, bid);
        let oid = hash_string(&elem.outputs()[0]);
        div.add_outputs(oid);
        div
    }

    pub fn add_inputs(&mut self, a: u64, b: u64) {
        self.a = a;
        self.b = b;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for DivNode<T> {
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

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(out) => {
                if let (
                    TypedArray::Float(a_arr),
                    TypedArray::Float(b_arr),
                    TypedArray::Float(o_arr),
                ) = (a, b, &mut *out)
                {
                    if a_arr.shape() == b_arr.shape() {
                        let a_sl = a_arr.as_slice_memory_order().unwrap();
                        let b_sl = b_arr.as_slice_memory_order().unwrap();
                        let dst = o_arr.as_slice_memory_order_mut().unwrap();
                        div_maybe_simd(a_sl, b_sl, dst);
                    } else {
                        a.div(b, out).unwrap();
                    }
                } else {
                    a.div(b, out).unwrap();
                }
            }
            _ => panic!("DivNode: missing input(s) - a={} b={}", self.a, self.b),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("div-{},{},{}", self.a, self.b, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [a, o] = omap.get_disjoint_mut([&self.a, &self.o]);
        let a = a.map(|arr| &*arr);
        if let (Some(a), Some(o)) = (a, o) {
            if let Some(in_shape) = a.shape() {
                *o = TypedArray::empty_with_others_type(a, in_shape).ensure_contiguous();
            }
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    impl_typed_binop!(div, /, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
}
