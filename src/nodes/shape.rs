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
pub struct ShapeNode<T: Default> {
    data: String,
    o: String,

    start: i64,
    end: Option<i64>,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ShapeNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut shape = Self {
            data: String::new(),
            o: String::new(),
            start: attrs.get("start").and_then(|v| v.as_int()).unwrap_or(0),
            end: attrs.get("end").and_then(|v| v.as_int()),
            unique_id: UniqueId::Shape,
            next_node: None,
        };
        shape.add_input_strings(elem.inputs[0].clone());
        shape.add_output_strings(elem.outputs[0].clone());
        Ok(shape)
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

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.data, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
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
            let len = end.saturating_sub(start);
            *o = TypedArray::Int64(ArrayD::zeros(IxDyn(&[len])));
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    pub fn shape_op(
        data: &TypedArray,
        start: i64,
        end: Option<i64>,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let shape: Vec<i64> = data
            .shape()
            .unwrap()
            .iter()
            .map(|val| *val as i64)
            .collect();

        let r = shape.len() as i64;

        let start = if start < 0 {
            (r + start).max(0) as usize
        } else {
            (start as usize).min(r as usize)
        };

        let end = match end {
            Some(e) => {
                if e < 0 {
                    (r + e).max(0) as usize
                } else {
                    (e as usize).min(r as usize)
                }
            }
            None => r as usize,
        };

        let sliced: Vec<i64> = if start >= end {
            vec![]
        } else {
            shape[start..end].to_vec()
        };

        let len = sliced.len();
        *o = TypedArray::Int64(ArrayD::from_shape_vec(IxDyn(&[len]), sliced).unwrap());

        Ok(())
    }
}
