use std::{any::Any, collections::HashMap};

use crate::{
    call_reshape_for_typed_array, get_curent_size_and_shape,
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::{AttributeValue, OnnxOperation};

#[derive(Default)]
pub struct ReshapeNode<T: Default> {
    data: String,
    shape: String,

    o: String,

    unique_id: UniqueId,

    allow_zero: bool,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ReshapeNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut reshape = Self {
            data: String::new(),
            shape: String::new(),

            o: String::new(),
            allow_zero: {
                match attrs.get("allow_zero") {
                    Some(av) => av.as_int().unwrap() != 0,
                    None => false,
                }
            },
            unique_id: UniqueId::Reshape,
            next_node: None,
        };
        reshape.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        reshape.add_output_strings(elem.outputs[0].clone());
        Ok(reshape)
    }
}

impl<T: Default> ReshapeNode<T> {
    pub fn new(allow_zero: bool) -> Self {
        Self {
            data: String::new(),
            shape: String::new(),

            o: String::new(),
            unique_id: UniqueId::Reshape,
            allow_zero,
            next_node: None,
        }
    }
    pub fn add_input_strings(&mut self, data: String, shape: String) {
        self.shape = shape;
        self.data = data;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ReshapeNode<T> {
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
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [data, shape, result] = omap.get_disjoint_mut([&self.data, &self.shape, &self.o]);
        let data = &*data.unwrap();
        let shape = &*shape.unwrap();

        match result {
            Some(result) => {
                data.reshape(shape, self.allow_zero, result).unwrap();
            }
            _ => panic!(
                "ReshapeNode: missing input(s) - data={} shape={}",
                self.data, self.shape
            ),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("reshape-{},{},{}", self.data, self.shape, self.o);
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
        let [data, shape, o] = omap.get_disjoint_mut([&self.data, &self.shape, &self.o]);
        let data = data.map(|arr| &*arr);
        let shape = shape.map(|arr| &*arr);

        if let (Some(data), Some(shape_tensor), Some(o)) = (data, shape, o)
            && let Some(in_shape) = data.shape()
            && let TypedArray::Int64(shape_arr) = shape_tensor
        {
            let current_size: usize = in_shape.iter().product();

            let mut new_shape: Vec<usize> = shape_arr
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if dim == -1 {
                        0
                    } else if dim == 0 {
                        if self.allow_zero {
                            0
                        } else {
                            *in_shape.get(i).unwrap_or(&0)
                        }
                    } else {
                        dim as usize
                    }
                })
                .collect();

            if let Some(idx) = shape_arr.iter().position(|&d| d == -1) {
                let known: usize = new_shape
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != idx)
                    .map(|(_, &d)| if d == 0 { 1 } else { d })
                    .product();
                new_shape[idx] = current_size / known;
            }

            *o = TypedArray::empty_with_others_type(data, &new_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    pub fn reshape(
        &self,
        shape: &TypedArray,
        allow_zero: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let shape_arr = match shape {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("reshape shape tensor must be I64")),
        };

        let (current_size, current_shape) = get_curent_size_and_shape!(
            self,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64, Bool
            ]
        );

        let mut new_shape: Vec<usize> = shape_arr
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if dim == -1 {
                    0
                } else if dim == 0 {
                    if allow_zero {
                        0
                    } else {
                        *current_shape.get(i).unwrap_or(&0)
                    }
                } else {
                    dim as usize
                }
            })
            .collect();

        if let Some(idx) = shape_arr.iter().position(|&d| d == -1) {
            let known: usize = new_shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &d)| if d == 0 { 1 } else { d })
                .product();
            new_shape[idx] = current_size / known;
        }

        call_reshape_for_typed_array!(
            self,
            new_shape,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }
}
