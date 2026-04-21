use std::any::Any;

use onnx_extractor::OnnxOperation;
use saker_rs::linarg::operations::apply_leaky_relu;

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Default)]
pub struct LeakyReluNode<T: Default> {
    pub x: String,

    alpha: f32,

    pub o: String,

    unique_id: UniqueId,

    pub next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for LeakyReluNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> anyhow::Result<Self> {
        let alpha = elem
            .attributes
            .get("alpha")
            .and_then(|val| val.as_float())
            .unwrap_or_else(|| 0.01f32);

        let mut leaky_relu = Self {
            x: String::new(),
            alpha,
            o: String::new(),
            unique_id: UniqueId::LeakyRelu,
            next_node: None,
        };
        leaky_relu.add_input_strings(elem.inputs[0].clone());
        leaky_relu.add_output_strings(elem.outputs[0].clone());

        Ok(leaky_relu)
    }
}

impl<T: Default> LeakyReluNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut leaky_relu = Self {
            x: String::new(),
            alpha: 0.01,
            o: String::new(),
            unique_id: UniqueId::LeakyRelu,
            next_node: None,
        };
        leaky_relu.add_input_strings(elem.inputs[0].clone());
        leaky_relu.add_output_strings(elem.outputs[0].clone());
        leaky_relu
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for LeakyReluNode<T> {
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
        println!("leaky relu-{},{}", self.x, self.o);
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
        let x = x.map(|inner| &*inner);

        match (x, o) {
            (Some(x), Some(result)) => {
                x.leaky_relu(self.alpha, result).unwrap();
            }
            _ => panic!("LeakyReluNode: missing input {}", self.x),
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
pub fn leaky_relu_f64(x: f64, alpha: f32) -> f64 {
    x.max(x * alpha as f64)
}

#[inline(always)]
pub fn leaky_relu_f32(x: f32, alpha: f32) -> f32 {
    x.max(x * alpha)
}

impl TypedArray {
    pub fn leaky_relu(&self, alpha: f32, o: &mut TypedArray) -> anyhow::Result<()> {
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;
        let in_shape = self.shape().ok_or_else(|| {
            anyhow::__private::must_use({
                let error = anyhow::__private::format_err(anyhow::__private::format_args!(
                    "undefined input"
                ));
                error
            })
        })?;
        match self {
            TypedArray::Float(_) => {
                let needs_alloc = match &*o {
                    TypedArray::Float(out) => out.shape() != in_shape,
                    _ => true,
                };
                if needs_alloc {
                    *o = TypedArray::empty_with_others_type(self, in_shape);
                }
            }
            TypedArray::Double(_) => {
                let needs_alloc = match &*o {
                    TypedArray::Double(out) => out.shape() != in_shape,
                    _ => true,
                };
                if needs_alloc {
                    *o = TypedArray::empty_with_others_type(self, in_shape);
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "{} only supported for given types",
                    stringify!($func_name)
                ));
            }
        }
        if let (TypedArray::Float(i), TypedArray::Float(o)) = (self, &mut *o) {
            let src = i.as_slice_memory_order().unwrap();
            let dst = o.as_slice_memory_order_mut().unwrap();
            apply_leaky_relu(dst, alpha, src);
            return Ok(());
        }
        match (self, &mut *o) {
            (TypedArray::Float(a), TypedArray::Float(o)) => {
                let src = a.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = leaky_relu_f32(*s, alpha));
            }
            (TypedArray::Double(a), TypedArray::Double(o)) => {
                let src = a.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = leaky_relu_f64(*s, alpha));
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "{} only supported for given types",
                    stringify!($func_name)
                ));
            }
        };
        Ok(())
    }
}
