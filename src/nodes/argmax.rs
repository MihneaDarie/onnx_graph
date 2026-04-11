use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, Axis, IxDyn};
use onnx_extractor::{AttributeValue, OnnxOperation};

#[derive(Default)]
pub struct ArgMaxNode<T: Default> {
    data: String,
    o: String,

    axis: i64,
    keepdims: bool,
    select_last_index: bool,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ArgMaxNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut argmax = Self {
            data: String::new(),
            o: String::new(),
            axis: attrs.get("axis").and_then(|v| v.as_int()).unwrap_or(0),
            keepdims: attrs.get("keepdims").and_then(|v| v.as_int()).unwrap_or(1) != 0,
            select_last_index: attrs
                .get("select_last_index")
                .and_then(|v| v.as_int())
                .unwrap_or(0)
                != 0,
            unique_id: UniqueId::ArgMax,
            next_node: None,
        };
        argmax.add_input_strings(elem.inputs[0].clone());
        argmax.add_output_strings(elem.outputs[0].clone());
        Ok(argmax)
    }
}

impl<T: Default> ArgMaxNode<T> {
    pub fn add_input_strings(&mut self, data: String) {
        self.data = data;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ArgMaxNode<T> {
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
        vec![self.data.clone()]
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [data, o] = omap.get_disjoint_mut([&self.data, &self.o]);
        let data = &*data.unwrap();

        match o {
            Some(result) => {
                TypedArray::argmax(
                    data,
                    self.axis,
                    self.keepdims,
                    self.select_last_index,
                    result,
                )
                .unwrap();
            }
            _ => panic!("ArgMaxNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "argmax-{},{} axis={} keepdims={} select_last={}",
            self.data, self.o, self.axis, self.keepdims, self.select_last_index
        );
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.data, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
            let ndim = in_shape.len() as i64;
            let axis = if self.axis < 0 {
                (ndim + self.axis) as usize
            } else {
                self.axis as usize
            };

            let mut out_shape: Vec<usize> = in_shape.to_vec();
            if self.keepdims {
                out_shape[axis] = 1;
            } else {
                out_shape.remove(axis);
            }

            *o = TypedArray::Int64(ArrayD::zeros(IxDyn(&out_shape))).ensure_contiguous();
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

macro_rules! call_argmax_for_typed_array {
    ($data:expr, $axis:expr, $keepdims:expr, $select_last_index:expr, $o:expr, [$($variant:ident),+]) => {

        match $data {
            $(
                TypedArray::$variant(arr) => argmax_variant!(arr, $axis, $keepdims, $select_last_index, $o),
            )+
            _ => return Err(anyhow::anyhow!("argmax: unsupported type")),
        }
    };
}

macro_rules! argmax_variant {
    ($arr:expr, $axis:expr, $keepdims:expr, $select_last_index:expr, $o:expr) => {{
        let ndim = $arr.ndim() as i64;
        let axis_usize = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let mut out_shape: Vec<usize> = $arr.shape().to_vec();
        let axis_len = out_shape[axis_usize];

        if $keepdims {
            out_shape[axis_usize] = 1;
        } else {
            out_shape.remove(axis_usize);
        }

        let needs_alloc = match &*($o) {
            TypedArray::Int64(out) => out.shape() != out_shape.as_slice(),
            _ => true,
        };

        if needs_alloc {
            *($o) = TypedArray::Int64(ArrayD::zeros(IxDyn(&out_shape)));
        }

        let out_arr = match $o {
            TypedArray::Int64(arr) => arr,
            _ => unreachable!(),
        };

        let out_sl = out_arr.as_slice_memory_order_mut().unwrap();
        let mut idx = 0;

        for lane in $arr.lanes(Axis(axis_usize)) {
            let mut max_val = lane[0];
            let mut max_idx: i64 = 0;

            for i in 1..axis_len {
                let val = lane[i];
                if $select_last_index {
                    if val >= max_val {
                        max_val = val;
                        max_idx = i as i64;
                    }
                } else {
                    if val > max_val {
                        max_val = val;
                        max_idx = i as i64;
                    }
                }
            }

            out_sl[idx] = max_idx;
            idx += 1;
        }
    }};
}

impl TypedArray {
    pub fn argmax(
        data: &TypedArray,
        axis: i64,
        keepdims: bool,
        select_last_index: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        call_argmax_for_typed_array!(
            data,
            axis,
            keepdims,
            select_last_index,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }
}
