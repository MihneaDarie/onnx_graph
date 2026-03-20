use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct TransposeNode<T: Default> {
    input: String,

    o: String,

    unique_id: UniqueId,

    perm: Vec<i64>,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromHashMap for TransposeNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            input: String::new(),
            o: String::new(),
            perm: match attrs.get("perm") {
                Some(av) => av.as_ints().unwrap().to_vec(),
                None => vec![],
            },
            unique_id: UniqueId::Transpose,
            next_node: None,
        })
    }
}

impl<T: Default> TransposeNode<T> {
    pub fn new(perm: Vec<i64>) -> Self {
        Self {
            input: String::new(),
            o: String::new(),
            perm,
            unique_id: UniqueId::Transpose,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, input: String) {
        self.input = input;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for TransposeNode<T> {
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

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.input.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.input, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                x.transpose(&self.perm, result).unwrap();
            }
            None => panic!("TransposeNode: missing input {}", self.input),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("transpose-{},{}", self.input, self.o);
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
        let [x, o] = omap.get_disjoint_mut([&self.input, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
            let ndim = in_shape.len() as i64;
            let perm: Vec<usize> = if self.perm.is_empty() {
                (0..in_shape.len()).rev().collect()
            } else {
                self.perm
                    .iter()
                    .map(|&p| {
                        if p < 0 {
                            (ndim + p) as usize
                        } else {
                            p as usize
                        }
                    })
                    .collect()
            };

            let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
            *o = TypedArray::empty_with_others_type(x, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
