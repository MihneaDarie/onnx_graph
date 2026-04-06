use std::{any::Any, collections::HashMap};

use crate::{
    call_pow_for_typed_array,
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::mul_maybe_simd;

#[derive(Default)]
pub struct PowNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> PowNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut pow = Self {
            a: String::new(),
            b: String::new(),
            o: String::new(),
            unique_id: UniqueId::Pow,
            next_node: None,
        };
        pow.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        pow.add_output_strings(elem.outputs[0].clone());
        pow
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for PowNode<T> {
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
                a.pow(b, out).unwrap();
            }
            _ => panic!("PowNode: missing output(s) - o={}", self.o),
        }
    }
    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }
    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("pow-{},{},{}", self.a, self.b, self.o);
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

impl TypedArray {
    pub fn pow(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        let in_shape = self.shape().unwrap();
        call_pow_for_typed_array!(
            self,
            b,
            o,
            in_shape,
            [(Float, f32), (Double, f64), (Int32, i32), (Int64, i64)]
        );
        Ok(())
    }
}
