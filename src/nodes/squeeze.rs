use std::any::Any;

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use onnx_extractor::OnnxOperation;

fn squeezed_shape(in_shape: &[usize], axes: Option<&[i64]>) -> anyhow::Result<Vec<usize>> {
    let rank = in_shape.len();

    let axes = match axes {
        Some(a) if !a.is_empty() => a,
        _ => {
            return Ok(in_shape.iter().copied().filter(|&d| d != 1).collect());
        }
    };

    let mut norm_axes: Vec<usize> = Vec::with_capacity(axes.len());
    for &a in axes {
        let a = if a < 0 { rank as i64 + a } else { a };
        if a < 0 || a as usize >= rank {
            anyhow::bail!("Squeeze: axis {a} out of range for rank {rank}");
        }
        let a = a as usize;
        if in_shape[a] != 1 {
            anyhow::bail!(
                "Squeeze: cannot squeeze axis {a} with dimension {} (must be 1)",
                in_shape[a]
            );
        }
        norm_axes.push(a);
    }
    norm_axes.sort_unstable();
    norm_axes.dedup();

    Ok(in_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !norm_axes.contains(i))
        .map(|(_, &d)| d)
        .collect())
}

#[derive(Default)]
pub struct SqueezeNode<T: Default> {
    data: u64,
    axes: Option<u64>,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> SqueezeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut squeeze = Self {
            data: u64::default(),
            axes: None,
            o: u64::default(),
            unique_id: UniqueId::Squeeze,
            next_node: None,
        };
        let inputs = elem
            .inputs()
            .iter()
            .filter(|name| !name.is_empty())
            .map(|val| hash_string(val))
            .collect::<Vec<u64>>();
        let o_id = hash_string(&elem.outputs()[0]);
        squeeze.add_inputs(&inputs);
        squeeze.add_outputs(o_id);
        squeeze
    }

    pub fn add_inputs(&mut self, inputs: &[u64]) {
        self.data = inputs[0];
        self.axes = inputs.get(1).copied();
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for SqueezeNode<T> {
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
        match self.axes {
            Some(axes_id) => {
                let [data, axes, o] = omap.get_disjoint_mut([&self.data, &axes_id, &self.o]);
                crate::debug_check_tensors!(
                    "SqueezeNode",
                    data => self.data,
                    axes => axes_id,
                    o => self.o,
                );
                let axes = axes.map(|val| &*val);
                if let (Some(data), Some(axes), Some(out)) = (data, axes, o) {
                    data.squeeze(Some(axes), out)?;
                }
            }
            None => {
                let [data, o] = omap.get_disjoint_mut([&self.data, &self.o]);
                crate::debug_check_tensors!(
                    "SqueezeNode",
                    data => self.data,
                    o => self.o,
                );
                if let (Some(data), Some(out)) = (data, o) {
                    data.squeeze(None, out)?;
                }
            }
        }
        Ok(())
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o]
    }

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.data]
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
        println!("squeeze-{},{}", self.data, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        match self.axes {
            Some(axes_id) => {
                let [x, axes, o] = omap.get_disjoint_mut([&self.data, &axes_id, &self.o]);
                let x = x.map(|arr| &*arr);
                let axes = axes.map(|arr| &*arr);

                if let (Some(x), Some(axes), Some(o)) = (x, axes, o)
                    && let Some(in_shape) = x.shape()
                    && let TypedArray::Int64(axes_arr) = axes
                {
                    let axes_vec: Vec<i64> = axes_arr.iter().copied().collect();
                    let out_shape = squeezed_shape(in_shape, Some(&axes_vec))?;
                    *o = TypedArray::empty_with_others_type(x, &out_shape);
                }
            }
            None => {
                let [x, o] = omap.get_disjoint_mut([&self.data, &self.o]);
                let x = x.map(|arr| &*arr);

                if let (Some(x), Some(o)) = (x, o)
                    && let Some(in_shape) = x.shape()
                {
                    let out_shape = squeezed_shape(in_shape, None)?;
                    *o = TypedArray::empty_with_others_type(x, &out_shape);
                }
            }
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
    pub fn squeeze(&self, axes: Option<&TypedArray>, o: &mut TypedArray) -> anyhow::Result<()> {
        let axes_vec: Option<Vec<i64>> = match axes {
            Some(TypedArray::Int64(a)) => Some(a.iter().copied().collect()),
            Some(_) => anyhow::bail!("Squeeze: axes must be I64"),
            None => None,
        };

        let in_shape = self
            .shape()
            .ok_or_else(|| anyhow::anyhow!("Squeeze: undefined input"))?;

        let out_shape = squeezed_shape(in_shape, axes_vec.as_deref())?;

        macro_rules! squeeze_variant {
            ($variant:ident, $a:expr) => {{
                use ndarray::ArrayD;
                use ndarray::IxDyn;

                let mut src_arr = $a.clone();
                let src = slice_memory_order_or_fix(&mut src_arr, "squeeze")?;
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
                        let dst = slice_memory_order_mut_or_fix(out, "squeeze")?;
                        dst.copy_from_slice(src);
                    }
                }
            }};
        }

        match self {
            TypedArray::Float(a) => squeeze_variant!(Float, a),
            TypedArray::Double(a) => squeeze_variant!(Double, a),
            TypedArray::Int32(a) => squeeze_variant!(Int32, a),
            TypedArray::Int64(a) => squeeze_variant!(Int64, a),
            TypedArray::Uint8(a) => squeeze_variant!(Uint8, a),
            TypedArray::Uint16(a) => squeeze_variant!(Uint16, a),
            TypedArray::Uint32(a) => squeeze_variant!(Uint32, a),
            TypedArray::Uint64(a) => squeeze_variant!(Uint64, a),
            TypedArray::Int8(a) => squeeze_variant!(Int8, a),
            TypedArray::Int16(a) => squeeze_variant!(Int16, a),
            TypedArray::Bool(a) => squeeze_variant!(Bool, a),
            _ => anyhow::bail!("Squeeze: unsupported type"),
        }

        Ok(())
    }
}