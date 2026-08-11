use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use ndarray::ArrayD;
use ndarray::IxDyn;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct RangeNode<T: Default> {
    start: u64,
    limit: u64,
    delta: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> RangeNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut range = Self {
            start: u64::default(),
            limit: u64::default(),
            delta: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Range,
            next_node: None,
        };
        let input_ids = elem
            .inputs()
            .iter()
            .map(|inp| hash_string(&inp))
            .collect::<Vec<u64>>();
        let o_id = hash_string(&elem.outputs()[0]);
        range.add_inputs(&input_ids);
        range.add_outputs(o_id);
        range
    }

    pub fn add_inputs(&mut self, inputs: &[u64]) {
        self.start = inputs[0].clone();
        self.limit = inputs[1].clone();
        self.delta = inputs[2].clone();
    }

    pub fn add_outputs(&mut self, o: u64) {
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

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [start, limit, delta, o] =
            omap.get_disjoint_mut([&self.start, &self.limit, &self.delta, &self.o]);
        crate::debug_check_tensors!(
            "RangeNode",
            start => self.start,
            limit => self.limit,
            delta => self.delta,
            o => self.o,
        );
        let start = start.map(|val| &*val);
        let limit = limit.map(|val| &*val);
        let delta = delta.map(|val| &*val);
        if let (Some(start), Some(limit), Some(delta), Some(out)) = (start, limit, delta, o) {
            TypedArray::range(start, limit, delta, out)?;
        }
        Ok(())
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn input_hashes(&self) -> Vec<u64> {
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

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [start, limit, delta, o] =
            omap.get_disjoint_mut([&self.start, &self.limit, &self.delta, &self.o]);
        let start = start.map(|inner| &*inner);
        let limit = limit.map(|inner| &*inner);
        let delta = delta.map(|inner| &*inner);

        if let (Some(start), Some(limit), Some(delta), Some(o)) = (start, limit, delta, o) {
            macro_rules! range_shape {
                ($variant:ident) => {
                    if let (
                        TypedArray::$variant(s),
                        TypedArray::$variant(l),
                        TypedArray::$variant(d),
                    ) = (start, limit, delta)
                    {
                        let s = *s.iter().next().ok_or_else(|| anyhow::anyhow!("Range: empty start"))?;
                        let l = *l.iter().next().ok_or_else(|| anyhow::anyhow!("Range: empty limit"))?;
                        let d = *d.iter().next().ok_or_else(|| anyhow::anyhow!("Range: empty delta"))?;
                        let n = (((l - s) as f64) / (d as f64)).ceil().max(0.0) as usize;
                        *o = TypedArray::$variant(ArrayD::zeros(IxDyn(&[n])));
                    }
                };
            }

            match start {
                TypedArray::Float(_) => range_shape!(Float),
                TypedArray::Double(_) => range_shape!(Double),
                TypedArray::Int32(_) => range_shape!(Int32),
                TypedArray::Int64(_) => range_shape!(Int64),
                TypedArray::Int16(_) => range_shape!(Int16),
                _ => {}
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
                    TypedArray::$variant(a) => *a
                        .iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Range: empty start"))?,
                    _ => anyhow::bail!("Range: start type mismatch"),
                };
                let l = match limit {
                    TypedArray::$variant(a) => *a
                        .iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Range: empty limit"))?,
                    _ => anyhow::bail!("Range: limit type mismatch"),
                };
                let d = match delta {
                    TypedArray::$variant(a) => *a
                        .iter()
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("Range: empty delta"))?,
                    _ => anyhow::bail!("Range: delta type mismatch"),
                };

                let n = (((l - s) as f64) / (d as f64)).ceil().max(0.0) as usize;

                let needs_alloc = match &*o {
                    TypedArray::$variant(out) => out.len() != n,
                    _ => true,
                };

                if needs_alloc {
                    let data: Vec<$T> = (0..n).map(|i| s + (i as $T) * d).collect();
                    *o = TypedArray::$variant(ArrayD::from_shape_vec(IxDyn(&[n]), data)?)
                        .ensure_contiguous();
                } else if let TypedArray::$variant(out) = o {
                    let dst = slice_memory_order_mut_or_fix(out, "range")?;
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
