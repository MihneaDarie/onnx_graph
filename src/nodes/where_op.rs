use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct WhereNode<T: Default> {
    c: String,
    x: String,
    y: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> WhereNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut where_op = Self {
            c: String::new(),
            x: String::new(),
            y: String::new(),
            o: String::new(),
            unique_id: UniqueId::Where,
            next_node: None,
        };
        where_op.add_input_strings(&elem.inputs);
        where_op.add_output_strings(elem.outputs[0].clone());
        where_op
    }

    pub fn broadcast_shape(shapes: &[&[usize]]) -> anyhow::Result<Vec<usize>> {
        let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut result = vec![1usize; max_rank];

        for shape in shapes {
            let offset = max_rank - shape.len();
            for (i, &dim) in shape.iter().enumerate() {
                let r = &mut result[offset + i];
                if *r == 1 {
                    *r = dim;
                } else if dim != 1 && dim != *r {
                    anyhow::bail!("Where: incompatible broadcast dimensions {} vs {}", *r, dim);
                }
            }
        }

        Ok(result)
    }

    pub fn add_input_strings(&mut self, inputs: &Vec<String>) {
        self.c = inputs[0].clone();
        self.x = inputs[1].clone();
        self.y = inputs[2].clone();
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for WhereNode<T> {
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
        let [c, x, y, o] = omap.get_disjoint_mut([&self.c, &self.x, &self.y, &self.o]);
        let c = c.map(|inner| &*inner);
        let x = x.map(|inner| &*inner);
        let y = y.map(|inner| &*inner);

        match (c, x, y, o) {
            (Some(c), Some(x), Some(y), Some(result)) => {
                TypedArray::where_op(c, x, y, result).unwrap();
            }
            _ => panic!("WhereNode: missing input {}", self.x),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.x.clone()]
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
        println!("where-{},{}", self.x, self.o);
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
        let [c, x, y, o] = omap.get_disjoint_mut([&self.c, &self.x, &self.y, &self.o]);
        let c = c.map(|arr| &*arr);
        let y = y.map(|arr| &*arr);
        let x = x.map(|arr| &*arr);
        if let (Some(c), Some(x), Some(y), Some(o)) = (c, x, y, o)
            && let (Some(c_shape), Some(x_shape), Some(y_shape)) = (c.shape(), x.shape(), y.shape())
        {
            let out_shape = Self::broadcast_shape(&[c_shape, x_shape, y_shape]).unwrap();
            *o = TypedArray::empty_with_others_type(x, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
