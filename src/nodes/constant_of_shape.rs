use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::{AttributeValue, OnnxOperation};

#[derive(Default)]
pub struct ConstantOfShapeNode<T: Default> {
    shape_array: String,

    value: Option<TypedArray>,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ConstantOfShapeNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let mut constant_of_shape = Self {
            shape_array: String::new(),
            value: None,
            o: String::new(),
            unique_id: UniqueId::ConstantOfShape,
            next_node: None,
        };

        let value = elem
            .attributes
            .get("value")
            .and_then(|val| AttributeValue::as_tensor(val))
            .and_then(|tensor| Some(TypedArray::from_tensor(&tensor)))
            .or_else(|| Some(TypedArray::Float(ArrayD::zeros(IxDyn(&[1])))));
        constant_of_shape.value = value;

        constant_of_shape.add_input_strings(elem.inputs[0].clone());
        constant_of_shape.add_output_strings(elem.outputs[0].clone());
        Ok(constant_of_shape)
    }
}

impl<T: Default> ConstantOfShapeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut constant_of_shape = Self {
            shape_array: String::new(),
            value: Some(TypedArray::Float(ArrayD::zeros(IxDyn(&[1])))),
            o: String::new(),
            unique_id: UniqueId::ConstantOfShape,
            next_node: None,
        };
        constant_of_shape.add_input_strings(elem.inputs[0].clone());
        constant_of_shape.add_output_strings(elem.outputs[0].clone());
        constant_of_shape
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.shape_array = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ConstantOfShapeNode<T> {
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
        let [x, o] = omap.get_disjoint_mut([&self.shape_array, &self.o]);
        let x = x.map(|inner| &*inner);

        match (x, o, &self.value) {
            (Some(x), Some(result), Some(value)) => {
                x.constant_of_shape(value, result).unwrap();
            }
            _ => panic!("ConstantOfShapeNode: missing input {}", self.shape_array),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.shape_array.clone()]
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
        println!("ConstantOfShape-{},{}", self.shape_array, self.o);
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
        let [x, o] = omap.get_disjoint_mut([&self.shape_array, &self.o]);
        let x = x.map(|inner| &*inner);

        match (x, o, &self.value) {
            (Some(x), Some(result), Some(value)) => {
                x.constant_of_shape(value, result).unwrap();
            }
            _ => panic!("ConstantOfShapeNode: missing input {}", self.shape_array),
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
