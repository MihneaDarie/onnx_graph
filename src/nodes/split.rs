use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::{AttributeValue, OnnxOperation};

#[derive(Default)]
pub struct SplitNode<T: Default> {
    input: String,
    split: String,

    o: Vec<String>,

    unique_id: UniqueId,

    axis: i64,
    num_outputs: i64,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for SplitNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut split = Self {
            input: String::new(),
            split: String::new(),

            o: vec![],

            unique_id: UniqueId::Split,

            axis: match attrs.get("axis") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            num_outputs: match attrs.get("num_outputs") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            next_node: None,
        };

        split.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        split.add_output_strings(elem.outputs.clone());

        Ok(split)
    }
}

impl<T: Default> SplitNode<T> {
    pub fn new(axis: i64, num_outputs: i64) -> Self {
        Self {
            input: String::new(),
            split: String::new(),

            o: vec![],
            axis,
            num_outputs,
            unique_id: UniqueId::Split,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, input: String, split: String) {
        self.input = input;
        self.split = split;
    }

    pub fn add_output_strings(&mut self, o: Vec<String>) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for SplitNode<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_unique_id(&self) -> UniqueId {
        self.unique_id
    }
    fn get_unique_id_mut(&mut self) -> UniqueId {
        self.unique_id
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let input = omap.get(&self.input);

        let split_sizes: Vec<i64> = if let Some(TypedArray::Int64(a)) = omap.get(&self.split) {
            a.iter().cloned().collect()
        } else if self.num_outputs > 0 {
            let input_ref = input.as_ref().unwrap();
            let axis = self.axis as usize;
            let dim = match input_ref {
                TypedArray::Float(a) => a.shape()[axis],
                _ => panic!("unsupported type"),
            };
            let chunk = dim / self.num_outputs as usize;
            vec![chunk as i64; self.num_outputs as usize]
        } else {
            panic!("SplitNode: no split tensor and no num_outputs");
        };

        match input {
            Some(input) => {
                let split_tensor = TypedArray::Int64(ndarray::Array1::from(split_sizes).into_dyn());
                let mut results = Vec::new();
                input.split(&split_tensor, self.axis, &mut results).unwrap();

                for (name, chunk) in self.o.iter().zip(results.into_iter()) {
                    omap.insert(name.clone(), chunk);
                }
            }
            None => panic!("SplitNode: missing input {}", self.input),
        }
    }

    fn output_names(&self) -> Vec<String> {
        self.o.clone()
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
        vec![self.input.clone(), self.split.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("split-{},{},{:?}", self.input, self.split, self.o);
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
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
