use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct ReduceMeanNode<T: Default> {
    data: String,
    axes: Option<String>,

    o: String,

    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ReduceMeanNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut reduce_mean = Self {
            data: String::new(),
            axes: None,
            keepdims: None,
            noop_with_empty_axes: None,
            o: String::new(),
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

        let inputs = &elem.inputs;
        let b = if inputs.len() == 2 {
            Some(inputs[1].clone())
        } else {
            None
        };
        reduce_mean.add_input_strings(inputs[0].clone(), b);
        reduce_mean.add_output_strings(elem.outputs[0].clone());
        Ok(reduce_mean)
    }
}

impl<T: Default> ReduceMeanNode<T> {
    pub fn add_input_strings(&mut self, a: String, b: Option<String>) {
        self.data = a;
        self.axes = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
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

    fn input_names(&self) -> Vec<String> {
        let mut names = vec![self.data.clone()];
        if let Some(axes) = &self.axes {
            names.push(axes.clone());
        }
        names
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let axes = &self.axes.clone().unwrap_or_default();
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &axes, &self.o]);
        let data = data.map(|inner| &*inner);
        let axes = axes.map(|inner| &*inner);

        match o {
            Some(result) => {}
            _ => panic!("ReduceMeanNode: missing output {}", self.o),
        }
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
            self.next_node = Some(vec![next]);
        }
        Ok(())
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let axes = &self.axes.clone().unwrap_or_default();
        let [data, axes, o] = omap.get_disjoint_mut([&self.data, &axes, &self.o]);
        let data = data.map(|inner| &*inner);
        let axes = axes.map(|inner| &*inner);

        if let Some(data) = data {
            let out_shape = {
                let in_shape = match data.shape() {
                    Some(s) => s.to_vec(),
                    None => return,
                };
                let ndim = in_shape.len();

                let axes_vec: Vec<usize> = match axes {
                    Some(TypedArray::Int64(ax)) if ax.len() > 0 => ax
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
                            return;
                        }
                        (0..ndim).collect()
                    }
                };

                let mut out_shape: Vec<usize> = Vec::new();
                for i in 0..ndim {
                    if axes_vec.contains(&i) {
                        if let Some(keepdims) = self.keepdims
                            && keepdims != 0
                        {
                            out_shape.push(1);
                        }
                    } else {
                        out_shape.push(in_shape[i]);
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

            if let Some(list) = &mut self.next_node {
                for next in list {
                    next.determine_output_shape(omap);
                }
            }
        }
    }
}

impl TypedArray {
    pub fn reduce_mean(
        &self,
        axes: Option<&TypedArray>,
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
                    Some(TypedArray::Int64(ax)) if ax.len() > 0 => ax
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
                                *o = TypedArray::$variant(ArrayD::zeros(IxDyn(in_shape))).ensure_contiguous();
                            }
                            if let TypedArray::$variant(out) = o {
                                let dst = out.as_slice_memory_order_mut().unwrap();
                                let src = $a.as_slice_memory_order().unwrap();
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
                    let dst = out.as_slice_memory_order_mut().unwrap();

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
                        let src = result_reshaped.as_slice_memory_order().unwrap();
                        dst.copy_from_slice(src);
                    } else {
                        let src = result.as_slice_memory_order().unwrap();
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
