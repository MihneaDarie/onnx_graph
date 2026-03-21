use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::mul_maybe_simd;

#[derive(Default)]
pub struct MulNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> MulNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut mul = Self {
            a: String::new(),
            b: String::new(),
            o: String::new(),
            unique_id: UniqueId::Mul,
            next_node: None,
        };
        mul.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        mul.add_output_strings(elem.outputs[0].clone());
        mul
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for MulNode<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }
    fn get_unique_id_mut(&mut self) -> UniqueId {
        self.unique_id
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.a.clone(), self.b.clone()]
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

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(out) => {
                if let (TypedArray::F32(a_arr), TypedArray::F32(b_arr), TypedArray::F32(o_arr)) =
                    (a, b, &mut *out)
                {
                    if a_arr.shape() == b_arr.shape() {
                        let a_sl = a_arr.as_slice_memory_order().unwrap();
                        let b_sl = b_arr.as_slice_memory_order().unwrap();
                        let dst = o_arr.as_slice_memory_order_mut().unwrap();
                        mul_maybe_simd(a_sl, b_sl, dst);
                    } else {
                        a.mul(b, out).unwrap();
                    }
                } else {
                    a.mul(b, out).unwrap();
                }
            }
            _ => panic!("MulNode: missing input(s) - a={} b={}", self.a, self.b),
        }
    }
    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }
    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("mul-{},{},{}", self.a, self.b, self.o);
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
