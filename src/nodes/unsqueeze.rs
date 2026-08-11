use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct UnsquezeeNode<T: Default> {
    data: u64,
    axes: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> UnsquezeeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut unsqueeze = Self {
            data: u64::default(),
            axes: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Unsqueeze,
            next_node: None,
        };
        let inputs = &elem
            .inputs()
            .iter()
            .map(|val| hash_string(val))
            .collect::<Vec<u64>>();
        let o_id = hash_string(&elem.outputs()[0]);
        unsqueeze.add_inputs(inputs);
        unsqueeze.add_outputs(o_id);
        unsqueeze
    }

    pub fn add_inputs(&mut self, inputs: &[u64]) {
        self.data = inputs[0].clone();
        self.axes = inputs[1].clone();
    }

    pub fn add_outputs(&mut self, o: u64) {
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

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &self.axes, &self.o]);
        crate::debug_check_tensors!(
            "UnsquezeeNode",
            data => self.data,
            axes => self.axes,
            o => self.o,
        );
        let axes = axes.map(|val| &*val);
        if let (Some(data), Some(axes), Some(out)) = (data, axes, o) {
            data.unsqueeze(axes, out)?;
        }
        Ok(())
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn input_hashes(&self) -> Vec<u64> {
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

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
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
                next.determine_output_shape(omap)?;
            }
        }
        Ok(())
    }
}

impl TypedArray {
    pub fn unsqueeze(&self, axes: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        let axes_vec: Vec<i64> = match axes {
            TypedArray::Int64(a) => a.iter().copied().collect(),
            _ => anyhow::bail!("Unsqueeze: axes must be I64"),
        };

        let in_shape = self
            .shape()
            .ok_or_else(|| anyhow::anyhow!("Unsqueeze: undefined input"))?;
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

        let mut out_shape: Vec<usize> = in_shape.to_vec();
        for &axis in norm_axes.iter() {
            out_shape.insert(axis, 1);
        }

        macro_rules! unsqueeze_variant {
            ($variant:ident, $a:expr) => {{
                use ndarray::ArrayD;
                use ndarray::IxDyn;

                let mut src_arr = $a.clone();
                let src = slice_memory_order_or_fix(&mut src_arr, "unsqueeze")?;
                let needs_realloc = match &*o {
                    TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };
                if needs_realloc {
                    *o = TypedArray::$variant(ArrayD::from_shape_vec(
                        IxDyn(&out_shape),
                        src.to_vec(),
                    )?)
                    .ensure_contiguous();
                } else {
                    if let TypedArray::$variant(out) = o {
                        let dst = slice_memory_order_mut_or_fix(out, "unsqueeze")?;
                        dst.copy_from_slice(src);
                    }
                }
            }};
        }

        match self {
            TypedArray::Float(a) => unsqueeze_variant!(Float, a),
            TypedArray::Double(a) => unsqueeze_variant!(Double, a),
            TypedArray::Int32(a) => unsqueeze_variant!(Int32, a),
            TypedArray::Int64(a) => unsqueeze_variant!(Int64, a),
            TypedArray::Uint8(a) => unsqueeze_variant!(Uint8, a),
            TypedArray::Uint16(a) => unsqueeze_variant!(Uint16, a),
            TypedArray::Uint32(a) => unsqueeze_variant!(Uint32, a),
            TypedArray::Uint64(a) => unsqueeze_variant!(Uint64, a),
            TypedArray::Int8(a) => unsqueeze_variant!(Int8, a),
            TypedArray::Int16(a) => unsqueeze_variant!(Int16, a),
            TypedArray::Bool(a) => unsqueeze_variant!(Bool, a),
            _ => anyhow::bail!("Unsqueeze: unsupported type"),
        }

        Ok(())
    }
}
