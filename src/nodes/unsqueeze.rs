use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct UnsquezeeNode<T: Default> {
    data: String,
    axes: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> UnsquezeeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut unsqueeze = Self {
            data: String::new(),
            axes: String::new(),
            o: String::new(),
            unique_id: UniqueId::Unsqueeze,
            next_node: None,
        };
        unsqueeze.add_input_strings(&elem.inputs);
        unsqueeze.add_output_strings(elem.outputs[0].clone());
        unsqueeze
    }

    pub fn add_input_strings(&mut self, inputs: &Vec<String>) {
        self.data = inputs[0].clone();
        self.axes = inputs[1].clone();
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for UnsquezeeNode<T> {
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
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &self.axes, &self.o]);
        let axes = axes.map(|val| &*val);
        let data = &*data.unwrap();

        match (axes, o) {
            (Some(axes), Some(result)) => {
                data.unsqueeze(axes, result).unwrap();
            }
            _ => panic!("UnsquezeeNode: missing input {}", self.data),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.data.clone()]
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

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("unsqueeze-{},{}", self.data, self.o);
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
        let [x, axes, o] = omap.get_disjoint_mut([&self.data, &self.axes, &self.o]);
        let x = x.map(|arr| &*arr);
        let axes = axes.map(|arr| &*arr);

        if let (Some(x), Some(axes), Some(o)) = (x, axes, o)
            && let Some(in_shape) = x.shape()
            && let TypedArray::Int64(axes_arr) = axes
        {
            let axes_vec: Vec<i64> = axes_arr.iter().copied().collect();
            let output_rank = in_shape.len() + axes_vec.len();

            let mut norm_axes: Vec<usize> = axes_vec
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (output_rank as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            norm_axes.sort();

            let mut out_shape = in_shape.to_vec();
            for &axis in &norm_axes {
                out_shape.insert(axis, 1);
            }

            *o = TypedArray::empty_with_others_type(x, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
