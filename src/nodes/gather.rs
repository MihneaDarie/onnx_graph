use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::{AttributeValue, OnnxOperation};

#[derive(Default)]
pub struct GatherNode<T: Default> {
    data: String,
    indices: String,
    o: String,

    axis: i64,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for GatherNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut gather = Self {
            data: String::new(),
            indices: String::new(),
            o: String::new(),
            axis: attrs.get("axis").and_then(|v| v.as_int()).unwrap_or(0),
            unique_id: UniqueId::Gather,
            next_node: None,
        };
        gather.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        gather.add_output_strings(elem.outputs[0].clone());
        Ok(gather)
    }
}

impl<T: Default> GatherNode<T> {
    pub fn add_input_strings(&mut self, data: String, indices: String) {
        self.data = data;
        self.indices = indices;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for GatherNode<T> {
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
        vec![self.data.clone(), self.indices.clone()]
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [data, indices, o] = omap.get_disjoint_mut([&self.data, &self.indices, &self.o]);
        let data = &*data.unwrap();
        let indices = &*indices.unwrap();

        match o {
            Some(result) => {
                TypedArray::gather(data, indices, self.axis, result).unwrap();
            }
            _ => panic!("GatherNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "gather-{},{},{} axis={}",
            self.data, self.indices, self.o, self.axis
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
        let [data, indices, o] = omap.get_disjoint_mut([&self.data, &self.indices, &self.o]);
        let data = data.map(|arr| &*arr);
        let indices = indices.map(|arr| &*arr);

        if let (Some(data), Some(indices), Some(o)) = (data, indices, o)
            && let (Some(data_shape), Some(idx_shape)) = (data.shape(), indices.shape())
        {
            let ndim = data_shape.len() as i64;
            let axis = if self.axis < 0 {
                (ndim + self.axis) as usize
            } else {
                self.axis as usize
            };

            let mut out_shape: Vec<usize> = Vec::new();
            for i in data_shape.iter().take(axis) {
                out_shape.push(*i);
            }
            for &s in idx_shape {
                out_shape.push(s);
            }
            for i in data_shape.iter().skip(axis + 1) {
                out_shape.push(*i);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }

            *o = TypedArray::empty_with_others_type(data, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
