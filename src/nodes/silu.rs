use std::any::Any;

use saker_rs::linarg::operations::apply_silu;

use crate::{
    call_activation_source_to_destination,
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Default)]
pub struct SiluNode<T: Default> {
    pub x: String,

    pub o: String,

    unique_id: UniqueId,

    pub next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> SiluNode<T> {
    pub fn new() -> Self {
        Self {
            x: String::new(),
            o: String::new(),
            unique_id: UniqueId::Silu,
            next_node: None,
        }
    }
}

impl<T: Default + 'static> Node<T> for SiluNode<T> {
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
        vec![self.x.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("silu-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                x.silu(result).unwrap();
            }
            None => panic!("SiluNode: missing input {}", self.x),
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
            *o = TypedArray::empty_with_others_type(x, in_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

#[inline(always)]
pub fn aprox_silu_f32(x: f32) -> f32 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn aprox_silu_f64(x: f64) -> f64 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

impl TypedArray {
    call_activation_source_to_destination!(
        silu,
        Some(apply_silu),
        [(Float, aprox_silu_f32), (Double, aprox_silu_f64)]
    );
}
