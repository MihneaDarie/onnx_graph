use std::{
    any::Any,
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::{
    impl_typed_binop,
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use ndarray::{ArrayD, IxDyn};
use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::add_maybe_simd;

#[derive(Default)]
pub struct AddNode<T: Default> {
    a: u64,
    b: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> AddNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut add = Self {
            a: u64::default(),
            b: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Add,
            next_node: None,
        };
        let aid = hash_string(&elem.inputs()[0]);
        let bid = hash_string(&elem.inputs()[1]);
        add.add_inputs(aid, bid);
        let oid = hash_string(&elem.outputs()[0]);
        add.add_outputs(oid);
        add
    }

    pub fn add_inputs(&mut self, a: u64, b: u64) {
        self.a = a;
        self.b = b;
    }

    pub fn add_outputs(&mut self, o: u64) {
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

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        crate::debug_check_tensors!("AddNode", a => self.a, b => self.b, o => self.o);
        if let (Some(a), Some(b), Some(out)) = (a, b, o) {
            if let (TypedArray::Float(a_arr), TypedArray::Float(b_arr)) = (&mut *a, &mut *b) {
                if a_arr.shape() == b_arr.shape() {
                    let needs_alloc = match &*out {
                        TypedArray::Float(o_arr) => o_arr.shape() != a_arr.shape(),
                        _ => true,
                    };
                    if needs_alloc {
                        *out = TypedArray::Float(ArrayD::zeros(IxDyn(a_arr.shape())));
                    }
                    if let TypedArray::Float(o_arr) = &mut *out {
                        let a_sl = slice_memory_order_or_fix(a_arr, "AddNode")?;
                        let b_sl = slice_memory_order_or_fix(b_arr, "AddNode")?;
                        let dst = slice_memory_order_mut_or_fix(o_arr, "AddNode")?;
                        add_maybe_simd(a_sl, b_sl, dst);
                    }
                } else {
                    a.add(b, out)?;
                }
            } else {
                a.add(b, out)?;
            }
        }
        Ok(())
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

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [a, o] = omap.get_disjoint_mut([&self.a, &self.o]);
        let a = a.map(|arr| &*arr);

        if let (Some(a), Some(o)) = (a, o)
            && let Some(in_shape) = a.shape()
        {
            *o = TypedArray::empty_with_others_type(a, in_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap)?;
            }
        }
        Ok(())
    }
}

impl TypedArray {
    impl_typed_binop!(add, +, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
}
