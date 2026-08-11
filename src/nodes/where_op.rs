use std::{
    any::Any,
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct WhereNode<T: Default> {
    c: u64,
    x: u64,
    y: u64,

    o: u64,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> WhereNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut where_op = Self {
            c: u64::default(),
            x: u64::default(),
            y: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Where,
            next_node: None,
        };
        where_op.add_inputs(
            &elem
                .inputs()
                .iter()
                .map(|inp| {
                    let mut hasher = DefaultHasher::new();
                    inp.hash(&mut hasher);
                    let id = hasher.finish();
                    id
                })
                .collect::<Vec<u64>>(),
        );
        let oid = hash_string(&elem.outputs()[0]);
        where_op.add_output(oid);
        where_op
    }

    pub fn broadcast_shape(shapes: &[&[usize]]) -> anyhow::Result<Vec<usize>> {
        let max_rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);

        if max_rank == 0 {
            return Ok(vec![]);
        }

        let mut result = vec![1usize; max_rank];

        for shape in shapes {
            if shape.is_empty() {
                continue;
            }
            let offset = max_rank - shape.len();
            for (i, &dim) in shape.iter().enumerate() {
                let r = &mut result[offset + i];
                if *r == 1 {
                    *r = dim;
                } else if dim != 1 && dim != *r {
                    anyhow::bail!("Where: incompatible broadcast dimensions {} vs {}", *r, dim);
                }
            }
        }

        Ok(result)
    }

    pub fn add_inputs(&mut self, inputs: &[u64]) {
        self.c = inputs[0].clone();
        self.x = inputs[1].clone();
        self.y = inputs[2].clone();
    }

    pub fn add_output(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for WhereNode<T> {
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
        let [c, x, y, o] = omap.get_disjoint_mut([&self.c, &self.x, &self.y, &self.o]);
        crate::debug_check_tensors!(
            "WhereNode",
            c => self.c,
            x => self.x,
            y => self.y,
            o => self.o,
        );
        let c = c.map(|val| &*val);
        let x = x.map(|val| &*val);
        let y = y.map(|val| &*val);
        if let (Some(c), Some(x), Some(y), Some(out)) = (c, x, y, o) {
            TypedArray::where_op(c, x, y, out)?;
        }
        Ok(())
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.x.clone()]
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
        println!("where-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [c, x, y, o] = omap.get_disjoint_mut([&self.c, &self.x, &self.y, &self.o]);
        let c = c.map(|arr| &*arr);
        let y = y.map(|arr| &*arr);
        let x = x.map(|arr| &*arr);

        if let (Some(c), Some(x), Some(y), Some(o)) = (c, x, y, o)
            && let (Some(c_shape), Some(x_shape), Some(y_shape)) = (c.shape(), x.shape(), y.shape())
        {
            match Self::broadcast_shape(&[c_shape, x_shape, y_shape]) {
                Ok(out_shape) => {
                    *o = TypedArray::empty_with_others_type(x, &out_shape);
                }
                Err(_) => {}
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

use ndarray::ArrayD;
use ndarray::IxDyn;

impl TypedArray {
    pub fn where_op(
        condition: &TypedArray,
        x: &TypedArray,
        y: &TypedArray,
        output: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let cond = match condition {
            TypedArray::Bool(c) => c,
            _ => anyhow::bail!("Where: condition must be Bool"),
        };

        let out_shape = WhereNode::<f32>::broadcast_shape(&[
            cond.shape(),
            x.shape().unwrap_or(&[]),
            y.shape().unwrap_or(&[]),
        ])?;

        macro_rules! ensure_alloc {
            ($variant:ident) => {{
                let needs_alloc = match &*output {
                    TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };
                if needs_alloc {
                    *output = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape)));
                }
            }};
        }

        match x {
            TypedArray::Float(_) => ensure_alloc!(Float),
            TypedArray::Double(_) => ensure_alloc!(Double),
            TypedArray::Int8(_) => ensure_alloc!(Int8),
            TypedArray::Int16(_) => ensure_alloc!(Int16),
            TypedArray::Int32(_) => ensure_alloc!(Int32),
            TypedArray::Int64(_) => ensure_alloc!(Int64),
            TypedArray::Uint8(_) => ensure_alloc!(Uint8),
            TypedArray::Uint16(_) => ensure_alloc!(Uint16),
            TypedArray::Uint32(_) => ensure_alloc!(Uint32),
            TypedArray::Uint64(_) => ensure_alloc!(Uint64),
            TypedArray::Bool(_) => {
                let needs_alloc = match &*output {
                    TypedArray::Bool(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };
                if needs_alloc {
                    *output = TypedArray::Bool(ArrayD::from_elem(IxDyn(&out_shape), false));
                }
            }
            _ => anyhow::bail!("Where: unsupported type"),
        }

        macro_rules! where_typed {
        ($(($variant:ident, $T:ty)),+) => {
            match (x, y, output) {
                $(
                    (
                        TypedArray::$variant(x_arr),
                        TypedArray::$variant(y_arr),
                        TypedArray::$variant(out_arr),
                    ) => {
                        let no_broadcast =
                            cond.shape() == out_shape.as_slice()
                            && x_arr.shape() == out_shape.as_slice()
                            && y_arr.shape() == out_shape.as_slice();

                        if no_broadcast {
                            let mut cond_arr = cond.clone();
                            let mut x_arr = x_arr.clone();
                            let mut y_arr = y_arr.clone();
                            let out_slice = slice_memory_order_mut_or_fix(out_arr, "where")?;
                            let c_slice = slice_memory_order_or_fix(&mut cond_arr, "where")?;
                            let x_slice = slice_memory_order_or_fix(&mut x_arr, "where")?;
                            let y_slice = slice_memory_order_or_fix(&mut y_arr, "where")?;

                            out_slice.iter_mut()
                                .zip(c_slice.iter())
                                .zip(x_slice.iter())
                                .zip(y_slice.iter())
                                .for_each(|(((o, c), xv), yv)| {
                                    *o = if *c { *xv } else { *yv };
                                });
                        } else {
                            let cond_b = cond
                                .broadcast(IxDyn(&out_shape))
                                .ok_or_else(|| anyhow::anyhow!("Where: condition broadcast failed"))?;
                            let x_b = x_arr
                                .broadcast(IxDyn(&out_shape))
                                .ok_or_else(|| anyhow::anyhow!("Where: x broadcast failed"))?;
                            let y_b = y_arr
                                .broadcast(IxDyn(&out_shape))
                                .ok_or_else(|| anyhow::anyhow!("Where: y broadcast failed"))?;

                            ndarray::Zip::from(out_arr)
                                .and(&cond_b)
                                .and(&x_b)
                                .and(&y_b)
                                .for_each(|o, c, xv, yv| {
                                    *o = if *c { *xv } else { *yv };
                                });
                        }
                        Ok(())
                    }
                )+
                _ => anyhow::bail!("Where: type mismatch between x, y, and output"),
            }
        };
    }

        where_typed!(
            (Float, f32),
            (Double, f64),
            (Int8, i8),
            (Int16, i16),
            (Int32, i32),
            (Int64, i64),
            (Uint8, u8),
            (Uint16, u16),
            (Uint32, u32),
            (Uint64, u64),
            (Bool, bool)
        )
    }
}
