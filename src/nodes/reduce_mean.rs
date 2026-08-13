use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct ReduceMeanNode<T: Default> {
    data: u64,
    axes: Option<u64>,
    
    axes_attr: Option<Vec<i64>>,

    o: u64,

    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ReduceMeanNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes();
        let mut reduce_mean = Self {
            data: u64::default(),
            axes: None,
            axes_attr: attrs
                .get("axes")
                .and_then(|val| val.as_ints())
                .map(|s| s.to_vec()),
            keepdims: None,
            noop_with_empty_axes: None,
            o: u64::default(),
            unique_id: UniqueId::Gemm,
            next_node: None,
        };

        reduce_mean.keepdims = attrs
            .get("keepdims")
            .and_then(|val| val.as_int())
            .or(Some(1));

        reduce_mean.noop_with_empty_axes = attrs
            .get("noop_with_empty_axes")
            .and_then(|val| val.as_int())
            .or(Some(0));

        let inputs = &elem.inputs();
        let axes_id = inputs.get(1).cloned().and_then(|val| {
            let id = hash_string(&val);
            Some(id)
        });
        let data_id = hash_string(&elem.inputs()[0]);
        let o_id = hash_string(&elem.outputs()[0]);
        reduce_mean.add_inputs(data_id, axes_id);
        reduce_mean.add_outputs(o_id);
        Ok(reduce_mean)
    }
}

impl<T: Default> ReduceMeanNode<T> {
    pub fn add_inputs(&mut self, a: u64, b: Option<u64>) {
        self.data = a;
        self.axes = b;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }

    /// Resolves the effective reduction axes, preferring the (opset 18+)
    /// tensor input when present, falling back to the (opset < 18) attribute.
    fn resolve_axes(&self, axes_tensor: Option<&TypedArray>) -> Option<Vec<i64>> {
        if let Some(TypedArray::Int64(ax)) = axes_tensor {
            if !ax.is_empty() {
                return Some(ax.iter().copied().collect());
            }
            return None;
        }
        self.axes_attr.clone()
    }
}

impl<T: Default + 'static> Node<T> for ReduceMeanNode<T> {
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

    fn input_hashes(&self) -> Vec<u64> {
        let mut names = vec![self.data.clone()];
        if let Some(axes) = &self.axes {
            names.push(axes.clone());
        }
        names
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let axes_key = self.axes.clone().unwrap_or_default();
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &axes_key, &self.o]);
        crate::debug_check_tensors!("ReduceMeanNode", data => self.data, o => self.o);
        if self.axes.is_some() {
            crate::debug_check_tensors!("ReduceMeanNode", axes => axes_key);
        }
        let axes_tensor = if self.axes.is_some() {
            axes.map(|val| &*val)
        } else {
            None
        };
        let resolved_axes = self.resolve_axes(axes_tensor);
        if let (Some(data), Some(out)) = (data, o) {
            let keepdims = self
                .keepdims
                .ok_or_else(|| anyhow::anyhow!("ReduceMeanNode: missing keepdims"))?;
            let noop = self
                .noop_with_empty_axes
                .ok_or_else(|| anyhow::anyhow!("ReduceMeanNode: missing noop_with_empty_axes"))?;
            data.reduce_mean(resolved_axes.as_deref(), keepdims != 0, noop != 0, out)?;
        }
        Ok(())
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("ReduceMean-{},{:?},{}", self.data, self.axes, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let axes_key = self.axes.clone().unwrap_or_default();
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &axes_key, &self.o]);
        let data = data.map(|inner| &*inner);
        let axes = axes.map(|inner| &*inner);
        let resolved_axes = self.resolve_axes(axes);

        if let Some(data) = data {
            let out_shape = {
                let in_shape = match data.shape() {
                    Some(s) => s.to_vec(),
                    None => {
                        if let Some(list) = &mut self.next_node {
                            for next in list {
                                next.determine_output_shape(omap)?;
                            }
                        }
                        return Ok(());
                    }
                };
                let ndim = in_shape.len();

                let axes_vec: Vec<usize> = match &resolved_axes {
                    Some(ax) if !ax.is_empty() => ax
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (ndim as i64 + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    _ => {
                        if let Some(noop_with_empty_axes) = self.noop_with_empty_axes
                            && noop_with_empty_axes != 0
                        {
                            if let Some(list) = &mut self.next_node {
                                for next in list {
                                    next.determine_output_shape(omap)?;
                                }
                            }
                            return Ok(());
                        }
                        (0..ndim).collect()
                    }
                };

                let mut out_shape: Vec<usize> = Vec::new();
                for (i, val) in in_shape.iter().enumerate().take(ndim) {
                    if axes_vec.contains(&i) {
                        if let Some(keepdims) = self.keepdims
                            && keepdims != 0
                        {
                            out_shape.push(1);
                        }
                    } else {
                        out_shape.push(*val);
                    }
                }
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                out_shape
            };
            if let Some(o) = o {
                *o = TypedArray::empty_with_others_type(data, &out_shape);
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
    pub fn reduce_mean(
        &self,
        axes: Option<&[i64]>,
        keepdims: bool,
        noop_with_empty_axes: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        macro_rules! reduce_mean_variant {
            ($variant:ident, $T:ty, $a:expr) => {{
                use ndarray::ArrayD;
                use ndarray::IxDyn;

                let in_shape = $a.shape();
                let ndim = in_shape.len();

                let axes_vec: Vec<usize> = match axes {
                    Some(ax) if !ax.is_empty() => ax
                        .iter()
                        .map(|&a| {
                            if a < 0 {
                                (ndim as i64 + a) as usize
                            } else {
                                a as usize
                            }
                        })
                        .collect(),
                    _ => {
                        if noop_with_empty_axes {
                            let needs_alloc = match &*o {
                                TypedArray::$variant(out) => out.shape() != in_shape,
                                _ => true,
                            };
                            if needs_alloc {
                                *o = TypedArray::$variant(ArrayD::zeros(IxDyn(in_shape)))
                                    .ensure_contiguous();
                            }
                            if let TypedArray::$variant(out) = o {
                                let mut src_arr = $a.clone();
                                let dst = slice_memory_order_mut_or_fix(out, "reduce_mean")?;
                                let src = slice_memory_order_or_fix(&mut src_arr, "reduce_mean")?;
                                dst.copy_from_slice(src);
                            }
                            return Ok(());
                        }
                        (0..ndim).collect()
                    }
                };

                let mut out_shape: Vec<usize> = Vec::new();
                for i in 0..ndim {
                    if axes_vec.contains(&i) {
                        if keepdims {
                            out_shape.push(1);
                        }
                    } else {
                        out_shape.push(in_shape[i]);
                    }
                }
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                let needs_alloc = match &*o {
                    TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };
                if needs_alloc {
                    *o = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape))).ensure_contiguous();
                }

                if let TypedArray::$variant(out) = o {
                    let dst = slice_memory_order_mut_or_fix(out, "reduce_mean")?;

                    let mut result = $a.clone();
                    let mut sorted_axes = axes_vec.clone();
                    sorted_axes.sort_by(|a, b| b.cmp(a));

                    for &axis in &sorted_axes {
                        result = result
                            .mean_axis(ndarray::Axis(axis))
                            .ok_or_else(|| anyhow::anyhow!("ReduceMean: mean_axis failed"))?;
                    }

                    if keepdims {
                        let result_reshaped = result.into_shape_with_order(IxDyn(&out_shape))?;
                        let mut src_arr = result_reshaped.clone();
                        let src = slice_memory_order_or_fix(&mut src_arr, "reduce_mean")?;
                        dst.copy_from_slice(src);
                    } else {
                        let mut src_arr = result.clone();
                        let src = slice_memory_order_or_fix(&mut src_arr, "reduce_mean")?;
                        dst[..src.len()].copy_from_slice(src);
                    }
                }
            }};
        }

        match self {
            TypedArray::Float(a) => reduce_mean_variant!(Float, f32, a),
            TypedArray::Double(a) => reduce_mean_variant!(Double, f64, a),
            TypedArray::Int32(a) => reduce_mean_variant!(Int32, i32, a),
            TypedArray::Int64(a) => reduce_mean_variant!(Int64, i64, a),
            TypedArray::Uint32(a) => reduce_mean_variant!(Uint32, u32, a),
            TypedArray::Uint64(a) => reduce_mean_variant!(Uint64, u64, a),
            _ => anyhow::bail!("ReduceMean: unsupported type"),
        }

        Ok(())
    }
}
