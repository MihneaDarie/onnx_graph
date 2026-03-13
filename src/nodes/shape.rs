use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::AttributeValue;

#[derive(Default)]
pub struct ShapeNode<T: Default> {
    data: String,
    o: String,

    start: i64,
    end: Option<i64>,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromHashMap for ShapeNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
            data: String::new(),
            o: String::new(),
            start: attrs.get("start").and_then(|v| v.as_int()).unwrap_or(0),
            end: attrs.get("end").and_then(|v| v.as_int()),
            unique_id: UniqueId::Shape,
            next_node: None,
        })
    }
}

impl<T: Default> ShapeNode<T> {
    pub fn add_input_strings(&mut self, data: String) {
        self.data = data;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ShapeNode<T> {
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
        vec![self.data.clone()]
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [data, o] = omap.get_disjoint_mut([&self.data, &self.o]);
        let data = &*data.unwrap();

        match o {
            Some(result) => {
                TypedArray::shape_op(data, self.start, self.end, result).unwrap();
            }
            _ => panic!("ShapeNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "shape-{},{} start={} end={:?}",
            self.data, self.o, self.start, self.end
        );
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
            self.next_node = Some(vec![next]);
        }
        Ok(())
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.data, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o) {
            if let Some(in_shape) = x.shape() {
                let r = in_shape.len() as i64;

                let start = if self.start < 0 {
                    (r + self.start).max(0) as usize
                } else {
                    (self.start as usize).min(r as usize)
                };

                let end = match self.end {
                    Some(e) => {
                        if e < 0 {
                            (r + e).max(0) as usize
                        } else {
                            (e as usize).min(r as usize)
                        }
                    }
                    None => r as usize,
                };

                let len = if start >= end { 0 } else { end - start };
                *o = TypedArray::I64(ArrayD::zeros(IxDyn(&[len])));
            }
        }
        
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
