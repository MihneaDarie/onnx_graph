use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct RangeNode<T: Default> {
    start: String,
    limit: String,
    delta: String,

    o: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> RangeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut range = Self {
            start: String::new(),
            limit: String::new(),
            delta: String::new(),
            o: String::new(),
            unique_id: UniqueId::Range,
            next_node: None,
        };
        range.add_input_strings(&elem.inputs);
        range.add_output_strings(elem.outputs[0].clone());
        range
    }

    pub fn add_input_strings(&mut self, inputs: &Vec<String>) {
        self.start = inputs[0].clone();
        self.limit = inputs[1].clone();
        self.delta = inputs[2].clone();
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for RangeNode<T> {
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
        let [start, limit, delta, o] =
            omap.get_disjoint_mut([&self.start, &self.limit, &self.delta, &self.o]);
        let start = start.map(|inner| &*inner);
        let limit = limit.map(|inner| &*inner);
        let delta = delta.map(|inner| &*inner);

        match (start, limit, delta, o) {
            (Some(start), Some(limit), Some(delta), Some(result)) => {
                TypedArray::range(start, limit, delta, result).unwrap();
            }
            _ => panic!("RangeNode: missing input {}", self.start),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.start.clone(), self.limit.clone(), self.delta.clone()]
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
        println!(
            "range-{},{}, {}, {}",
            self.start, self.limit, self.delta, self.o
        );
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
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

impl TypedArray {
    pub fn range(
        start: &TypedArray,
        limit: &TypedArray,
        delta: &TypedArray,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        macro_rules! range_variant {
            ($variant:ident, $T:ty) => {{
                use ndarray::ArrayD;
                use ndarray::IxDyn;
                let s = match start {
                    TypedArray::$variant(a) => *a.iter().next().unwrap(),
                    _ => anyhow::bail!("Range: start type mismatch"),
                };
                let l = match limit {
                    TypedArray::$variant(a) => *a.iter().next().unwrap(),
                    _ => anyhow::bail!("Range: limit type mismatch"),
                };
                let d = match delta {
                    TypedArray::$variant(a) => *a.iter().next().unwrap(),
                    _ => anyhow::bail!("Range: delta type mismatch"),
                };

                let n = (((l - s) as f64) / (d as f64)).ceil().max(0.0) as usize;

                let needs_alloc = match &*o {
                    TypedArray::$variant(out) => out.len() != n,
                    _ => true,
                };

                if needs_alloc {
                    let data: Vec<$T> = (0..n).map(|i| s + (i as $T) * d).collect();
                    *o = TypedArray::$variant(ArrayD::from_shape_vec(IxDyn(&[n]), data)?).ensure_contiguous();
                } else if let TypedArray::$variant(out) = o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    for i in 0..n {
                        dst[i] = s + (i as $T) * d;
                    }
                }
            }};
        }

        match start {
            TypedArray::Float(_) => range_variant!(Float, f32),
            TypedArray::Double(_) => range_variant!(Double, f64),
            TypedArray::Int16(_) => range_variant!(Int16, i16),
            TypedArray::Int32(_) => range_variant!(Int32, i32),
            TypedArray::Int64(_) => range_variant!(Int64, i64),
            _ => anyhow::bail!("Range: unsupported type"),
        }

        Ok(())
    }
}
