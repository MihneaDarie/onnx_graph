use std::any::Any;

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct GatherNode<T: Default> {
    data: String,
    indices: String,
    o: String,

    axis: i64,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for GatherNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut gather = Self {
            data: String::new(),
            indices: String::new(),
            o: String::new(),
            axis: attrs.get("axis").and_then(|v| v.as_int()).unwrap_or(0),
            unique_id: UniqueId::Gather,
            next_node: None,
        };
        gather.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        gather.add_output_strings(elem.outputs[0].clone());
        Ok(gather)
    }
}

impl<T: Default> GatherNode<T> {
    pub fn add_input_strings(&mut self, data: String, indices: String) {
        self.data = data;
        self.indices = indices;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for GatherNode<T> {
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
        vec![self.data.clone(), self.indices.clone()]
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [data, indices, o] = omap.get_disjoint_mut([&self.data, &self.indices, &self.o]);
        let data = &*data.unwrap();
        let indices = &*indices.unwrap();

        match o {
            Some(result) => {
                TypedArray::gather(data, indices, self.axis, result).unwrap();
            }
            _ => panic!("GatherNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "gather-{},{},{} axis={}",
            self.data, self.indices, self.o, self.axis
        );
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [data, indices, o] = omap.get_disjoint_mut([&self.data, &self.indices, &self.o]);
        let data = data.map(|arr| &*arr);
        let indices = indices.map(|arr| &*arr);

        if let (Some(data), Some(indices), Some(o)) = (data, indices, o)
            && let (Some(data_shape), Some(idx_shape)) = (data.shape(), indices.shape())
        {
            let ndim = data_shape.len() as i64;
            let axis = if self.axis < 0 {
                (ndim + self.axis) as usize
            } else {
                self.axis as usize
            };

            let is_scalar_index = idx_shape.is_empty() || idx_shape == [1];

            let mut out_shape: Vec<usize> = Vec::new();
            for i in data_shape.iter().take(axis) {
                out_shape.push(*i);
            }
            if !is_scalar_index {
                for &s in idx_shape {
                    out_shape.push(s);
                }
            }
            for i in data_shape.iter().skip(axis + 1) {
                out_shape.push(*i);
            }

            *o = TypedArray::empty_with_others_type(data, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

macro_rules! call_gather_for_typed_array {
    ($data:expr, $axis:expr, $idx_vec:expr, $idx_shape:expr, $o:expr, [$($variant:ident),+]) => {

        match $data {
            $(
                TypedArray::$variant(arr) => gather_variant!($variant, $axis, $idx_vec, $idx_shape, arr, $o),
            )+
            TypedArray::Bool(arr) => {
                let ndim = arr.ndim() as i64;
                let axis_usize = if $axis < 0 {
                    (ndim + $axis) as usize
                } else {
                    $axis as usize
                };

                let data_shape = arr.shape();
                let axis_size = data_shape[axis_usize] as i64;

                let is_scalar_index = $idx_shape.is_empty() || $idx_shape == [1usize];

                let mut out_shape: Vec<usize> = Vec::new();
                for i in 0..axis_usize {
                    out_shape.push(data_shape[i]);
                }
                if !is_scalar_index {
                    for &s in &($idx_shape) {
                        out_shape.push(s);
                    }
                }
                for i in (axis_usize + 1)..data_shape.len() {
                    out_shape.push(data_shape[i]);
                }


                let needs_alloc = match &*($o) {
                    TypedArray::Bool(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };

                if needs_alloc {
                    *($o) = TypedArray::Bool(ArrayD::from_elem(IxDyn(&out_shape),false));
                }

                let out_arr = match $o {
                    TypedArray::Bool(arr) => arr,
                    _ => unreachable!(),
                };

                let data_sl = arr.as_slice_memory_order().unwrap();
                let out_sl = out_arr.as_slice_memory_order_mut().unwrap();

                let outer_size: usize = data_shape[..axis_usize].iter().product();
                let inner_size: usize = data_shape[axis_usize + 1..].iter().product();
                let axis_dim = data_shape[axis_usize];

                let mut out_idx = 0;
                for outer in 0..outer_size.max(1) {
                    for &idx in &($idx_vec) {
                        let idx = if idx < 0 {
                            (axis_size + idx) as usize
                        } else {
                            idx as usize
                        };

                        let src_offset = outer * axis_dim * inner_size + idx * inner_size;
                        let len = inner_size.max(1);

                        out_sl[out_idx..out_idx + len]
                            .copy_from_slice(&data_sl[src_offset..src_offset + len]);
                        out_idx += len;
                    }
                }
            }
            _ => return Err(anyhow::anyhow!("argmax: unsupported type")),
        }
    };
}

macro_rules! gather_variant {
    ($variant:ident, $axis:expr, $idx_vec:expr, $idx_shape:expr, $arr:expr, $o:expr) => {{
        let ndim = $arr.ndim() as i64;
        let axis_usize = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let data_shape = $arr.shape();
        let axis_size = data_shape[axis_usize] as i64;

        let is_scalar_index = $idx_shape.is_empty() || $idx_shape == [1usize];

        let mut out_shape: Vec<usize> = Vec::new();
        for i in 0..axis_usize {
            out_shape.push(data_shape[i]);
        }
        if !is_scalar_index {
            for &s in &($idx_shape) {
                out_shape.push(s);
            }
        }
        for i in (axis_usize + 1)..data_shape.len() {
            out_shape.push(data_shape[i]);
        }

        let needs_alloc = match &*($o) {
            TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
            _ => true,
        };

        if needs_alloc {
            *($o) = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape)));
        }

        let out_arr = match $o {
            TypedArray::$variant(arr) => arr,
            _ => unreachable!(),
        };

        let data_sl = $arr.as_slice_memory_order().unwrap();
        let out_sl = out_arr.as_slice_memory_order_mut().unwrap();

        let outer_size: usize = data_shape[..axis_usize].iter().product();
        let inner_size: usize = data_shape[axis_usize + 1..].iter().product();
        let axis_dim = data_shape[axis_usize];

        let mut out_idx = 0;
        for outer in 0..outer_size.max(1) {
            for &idx in &($idx_vec) {
                let idx = if idx < 0 {
                    (axis_size + idx) as usize
                } else {
                    idx as usize
                };

                let src_offset = outer * axis_dim * inner_size + idx * inner_size;
                let len = inner_size.max(1);

                out_sl[out_idx..out_idx + len]
                    .copy_from_slice(&data_sl[src_offset..src_offset + len]);
                out_idx += len;
            }
        }
    }};
}

impl TypedArray {
    pub fn gather(
        data: &TypedArray,
        indices: &TypedArray,
        axis: i64,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let idx_vec: Vec<i64> = match indices {
            TypedArray::Int64(arr) => arr.iter().copied().collect(),
            TypedArray::Int32(arr) => arr.iter().map(|&v| v as i64).collect(),
            _ => return Err(anyhow::anyhow!("Gather: indices must be I32 or I64")),
        };
        let idx_shape: Vec<usize> = match indices {
            TypedArray::Int64(arr) => arr.shape().to_vec(),
            TypedArray::Int32(arr) => arr.shape().to_vec(),
            _ => unreachable!(),
        };

        call_gather_for_typed_array!(
            data,
            axis,
            idx_vec,
            idx_shape,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }
}
