use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct ExpandNode<T: Default> {
    input: String,
    shape: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> ExpandNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut expand = Self {
            input: String::new(),
            shape: String::new(),
            o: String::new(),
            unique_id: UniqueId::Expand,
            next_node: None,
        };
        expand.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        expand.add_output_strings(elem.outputs[0].clone());
        expand
    }

    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.input = a;
        self.shape = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ExpandNode<T> {
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

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.input.clone(), self.shape.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.input, &self.shape, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(out) => {}
            _ => panic!(
                "ExpandNode: missing input(s) - a={} b={}",
                self.input, self.shape
            ),
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

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("expand-{},{},{}", self.input, self.shape, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
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
        let [input, shape, o] = omap.get_disjoint_mut([&self.input, &self.shape, &self.o]);
        let input = input.map(|inner| &*inner);
        let shape = shape.map(|inner| &*inner);

        if let (Some(input), Some(shape), Some(o)) = (input, shape, o)
            && let Some(in_shape) = input.shape()
            && let TypedArray::Int64(target_arr) = shape
        {
            let target_shape: Vec<usize> = target_arr.iter().map(|&v| v as usize).collect();

            let out_rank = in_shape.len().max(target_shape.len());
            let mut out_shape = vec![0usize; out_rank];

            for i in 0..out_rank {
                let in_dim = if i < out_rank - in_shape.len() {
                    1
                } else {
                    in_shape[i - (out_rank - in_shape.len())]
                };
                let target_dim = if i < out_rank - target_shape.len() {
                    1
                } else {
                    target_shape[i - (out_rank - target_shape.len())]
                };

                out_shape[i] = if in_dim == 1 {
                    target_dim
                } else if target_dim == 1 {
                    in_dim
                } else {
                    in_dim
                };
            }

            *o = TypedArray::empty_with_others_type(input, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    pub fn expand(&self, shape_tensor: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        let target_shape: Vec<usize> = match shape_tensor {
            TypedArray::Int64(s) => s.iter().map(|&v| v as usize).collect(),
            _ => anyhow::bail!("Expand: shape must be I64"),
        };

        let in_shape = self
            .shape()
            .ok_or_else(|| anyhow::anyhow!("Expand: undefined input"))?;

        let out_rank = in_shape.len().max(target_shape.len());
        let mut out_shape = vec![0usize; out_rank];

        for i in 0..out_rank {
            let in_dim = if i < out_rank - in_shape.len() {
                1
            } else {
                in_shape[i - (out_rank - in_shape.len())]
            };
            let target_dim = if i < out_rank - target_shape.len() {
                1
            } else {
                target_shape[i - (out_rank - target_shape.len())]
            };

            if in_dim == target_dim {
                out_shape[i] = in_dim;
            } else if in_dim == 1 {
                out_shape[i] = target_dim;
            } else if target_dim == 1 {
                out_shape[i] = in_dim;
            } else {
                anyhow::bail!(
                    "Expand: incompatible shapes {:?} and {:?} at dim {}",
                    in_shape,
                    target_shape,
                    i
                );
            }
        }

        macro_rules! expand_variant {
            ($variant:ident, $a:expr) => {{
                use ndarray::IxDyn;
                let needs_alloc = match &*o {
                    TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };

                if needs_alloc {
                    *o = TypedArray::empty_with_others_type(self, &out_shape);
                }

                if let TypedArray::$variant(out) = o {
                    let view = $a
                        .broadcast(IxDyn(&out_shape))
                        .ok_or_else(|| anyhow::anyhow!("Expand: broadcast failed"))?;

                    let dst = out.as_slice_memory_order_mut().unwrap();
                    for (d, s) in dst.iter_mut().zip(view.iter()) {
                        *d = *s;
                    }
                }
            }};
        }

        match self {
            TypedArray::Float(a) => expand_variant!(Float, a),
            TypedArray::Double(a) => expand_variant!(Double, a),
            TypedArray::Int8(a) => expand_variant!(Int8, a),
            TypedArray::Int16(a) => expand_variant!(Int16, a),
            TypedArray::Int32(a) => expand_variant!(Int32, a),
            TypedArray::Int64(a) => expand_variant!(Int64, a),
            TypedArray::Uint8(a) => expand_variant!(Uint8, a),
            TypedArray::Uint16(a) => expand_variant!(Uint16, a),
            TypedArray::Uint32(a) => expand_variant!(Uint32, a),
            TypedArray::Uint64(a) => expand_variant!(Uint64, a),
            TypedArray::Bool(a) => expand_variant!(Bool, a),
            _ => anyhow::bail!("Expand: unsupported type"),
        }

        Ok(())
    }
}
