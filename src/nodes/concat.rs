use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ConcatNode<T> {
    inputs: Vec<String>,

    o: String,

    unique_id: UniqueId,

    axis: i64,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> ConcatNode<T> {
    pub fn add_input_strings(&mut self, inputs: Vec<String>) {
        self.inputs = inputs;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default> FromHashMap for ConcatNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            axis: {
                match attrs.get("axis") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            next_node: None,
            inputs: vec![],
            o: String::new(),
            unique_id: UniqueId::Concat,
        })
    }
}

impl<T: Default + 'static> Node<T> for ConcatNode<T> {
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
        vec![self.o.clone()]
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

    fn output_names(&self) -> Vec<String> {
        self.inputs.clone()
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let arrays: Vec<&TypedArray> = self
            .inputs
            .iter()
            .map(|name| {
                omap.get(name)
                    .unwrap_or_else(|| panic!("ConcatNode: missing input {}", name))
            })
            .collect();

        let ndim = match &arrays[0] {
            TypedArray::F32(a) => a.ndim(),
            TypedArray::F64(a) => a.ndim(),
            TypedArray::I32(a) => a.ndim(),
            TypedArray::I64(a) => a.ndim(),
            _ => panic!("unsupported type in concat"),
        };

        let axis = if self.axis < 0 {
            (ndim as i64 + self.axis) as usize
        } else {
            self.axis as usize
        };

        let refs: Vec<&TypedArray> = arrays;
        let mut result = TypedArray::Undefined;
        TypedArray::concat(&refs, axis, &mut result).unwrap();
        omap.insert(self.o.clone(), result);
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("concat-{:?},{}", self.inputs, self.o);
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
}
