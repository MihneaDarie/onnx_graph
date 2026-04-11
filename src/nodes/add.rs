use std::any::Any;

use crate::{
    impl_typed_binop,
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use ndarray::{ArrayD, IxDyn};
use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::add_maybe_simd;

#[derive(Default)]
pub struct AddNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> AddNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut add = Self {
            a: String::new(),
            b: String::new(),
            o: String::new(),
            unique_id: UniqueId::Add,
            next_node: None,
        };
        add.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        add.add_output_strings(elem.outputs[0].clone());
        add
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for AddNode<T> {
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

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
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
                if let (TypedArray::Float(a_arr), TypedArray::Float(b_arr)) = (a, b) {
                    if a_arr.shape() == b_arr.shape() {
                        let needs_alloc = match &*out {
                            TypedArray::Float(o_arr) => o_arr.shape() != a_arr.shape(),
                            _ => true,
                        };
                        if needs_alloc {
                            *out = TypedArray::Float(ArrayD::zeros(IxDyn(a_arr.shape())));
                        }
                        if let TypedArray::Float(o_arr) = &mut *out {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = o_arr.as_slice_memory_order_mut().unwrap();
                            add_maybe_simd(a_sl, b_sl, dst);
                        }
                    } else {
                        a.add(b, out).unwrap();
                    }
                } else {
                    a.add(b, out).unwrap();
                }
            }
            _ => panic!("AddNode: missing output"),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("add-{},{},{}", self.a, self.b, self.o);
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
            *o = TypedArray::empty_with_others_type(a, in_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    impl_typed_binop!(add, +, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
}
