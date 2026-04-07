use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::{AttributeValue, OnnxOperation};
use saker_rs::{activations::Activation, linarg::operations::sgemm_bias_parallel};

#[derive(Default)]
pub struct MatMulNode<T: Default> {
    a: String,
    b: String,

    o: String,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for MatMulNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut gemm = Self {
            a: String::new(),
            b: String::new(),

            o: String::new(),
            unique_id: UniqueId::Gemm,
            next_node: None,
        };
        let inputs = &elem.inputs;
        gemm.add_input_strings(inputs[0].clone(), inputs[1].clone());
        gemm.add_output_strings(elem.outputs[0].clone());
        Ok(gemm)
    }
}

impl<T: Default> MatMulNode<T> {
    pub fn add_input_strings(&mut self, a: String, b: String) {
        self.a = a;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for MatMulNode<T> {
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
        let names = vec![self.a.clone(), self.b.clone()];
        names
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = &*a.unwrap();
        let b = &*b.unwrap();

        match o {
            Some(result) => {}
            _ => panic!("MatMulNode: missing output {}", self.o),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("MatMul-{},{:?},{}", self.a, self.b, self.o);
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
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = a.map(|inner| &*inner);
        let b = b.map(|inner| &*inner);
        if let (Some(a), Some(b)) = (a, b) {
            let out_shape = {
                let a_shape = match a.shape() {
                    Some(s) => s.to_vec(),
                    None => return,
                };
                let b_shape = match b.shape() {
                    Some(s) => s.to_vec(),
                    None => return,
                };
                let a_ndim = a_shape.len();
                let b_ndim = b_shape.len();

                let out_shape = match (a_ndim, b_ndim) {
                    (1, 1) => vec![1],
                    (2, 1) => vec![a_shape[0]],
                    (1, 2) => vec![b_shape[1]],
                    (2, 2) => vec![a_shape[0], b_shape[1]],
                    _ => {
                        let m = a_shape[a_ndim - 2];
                        let n = b_shape[b_ndim - 1];
                        let a_batch = &a_shape[..a_ndim - 2];
                        let b_batch = &b_shape[..b_ndim - 2];
                        let batch_rank = a_batch.len().max(b_batch.len());

                        let mut batch_shape = vec![0usize; batch_rank];
                        for i in 0..batch_rank {
                            let a_dim = if i < batch_rank - a_batch.len() {
                                1
                            } else {
                                a_batch[i - (batch_rank - a_batch.len())]
                            };
                            let b_dim = if i < batch_rank - b_batch.len() {
                                1
                            } else {
                                b_batch[i - (batch_rank - b_batch.len())]
                            };
                            batch_shape[i] = a_dim.max(b_dim);
                        }
                        let mut shape = batch_shape;
                        shape.push(m);
                        shape.push(n);
                        shape
                    }
                };

                out_shape
            };

            if let Some(o) = o {
                *o = TypedArray::empty_with_others_type(a, &out_shape);
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
    pub fn matmul(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, b) {
            (TypedArray::Float(a_arr), TypedArray::Float(b_arr)) => {
                let a_shape = a_arr.shape();
                let b_shape = b_arr.shape();
                let a_ndim = a_shape.len();
                let b_ndim = b_shape.len();

                match (a_ndim, b_ndim) {
                    (1, 1) => {
                        let k = a_shape[0];
                        assert_eq!(k, b_shape[0]);
                        let needs_alloc = match &*o {
                            TypedArray::Float(out) => out.shape() != &[1],
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&[1]))).ensure_contiguous();
                        }
                        if let TypedArray::Float(out) = o {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            dst[0] = a_sl.iter().zip(b_sl.iter()).map(|(a, b)| a * b).sum();
                        }
                    }

                    (2, 1) => {
                        let m = a_shape[0];
                        let k = a_shape[1];
                        assert_eq!(k, b_shape[0]);
                        let out_shape = [m];
                        let needs_alloc = match &*o {
                            TypedArray::Float(out) => out.shape() != out_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&out_shape)))
                                .ensure_contiguous();
                        }
                        if let TypedArray::Float(out) = o {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            for i in 0..m {
                                let mut sum = 0.0f32;
                                for p in 0..k {
                                    sum += a_sl[i * k + p] * b_sl[p];
                                }
                                dst[i] = sum;
                            }
                        }
                    }

                    (1, 2) => {
                        let k = a_shape[0];
                        let n = b_shape[1];
                        assert_eq!(k, b_shape[0]);
                        let out_shape = [n];
                        let needs_alloc = match &*o {
                            TypedArray::Float(out) => out.shape() != out_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&out_shape)))
                                .ensure_contiguous();
                        }
                        if let TypedArray::Float(out) = o {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = out.as_slice_memory_order_mut().unwrap();
                            for j in 0..n {
                                let mut sum = 0.0f32;
                                for p in 0..k {
                                    sum += a_sl[p] * b_sl[p * n + j];
                                }
                                dst[j] = sum;
                            }
                        }
                    }

                    (2, 2) => {
                        let m = a_shape[0];
                        let k = a_shape[1];
                        let n = b_shape[1];
                        assert_eq!(k, b_shape[0]);

                        let out_shape = [m, n];
                        let needs_alloc = match &*o {
                            TypedArray::Float(out) => out.shape() != out_shape,
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&out_shape)))
                                .ensure_contiguous();
                        }
                        if let TypedArray::Float(out) = o {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = out.as_slice_memory_order_mut().unwrap();

                            sgemm_bias_parallel(m, n, k, a_sl, b_sl, None, dst, Activation::None);
                        }
                    }

                    _ => {
                        let m = a_shape[a_ndim - 2];
                        let k = a_shape[a_ndim - 1];
                        let n = b_shape[b_ndim - 1];
                        assert_eq!(k, b_shape[b_ndim - 2]);

                        let a_batch = &a_shape[..a_ndim - 2];
                        let b_batch = &b_shape[..b_ndim - 2];
                        let batch_rank = a_batch.len().max(b_batch.len());

                        let mut batch_shape = vec![0usize; batch_rank];
                        for i in 0..batch_rank {
                            let a_dim = if i < batch_rank - a_batch.len() {
                                1
                            } else {
                                a_batch[i - (batch_rank - a_batch.len())]
                            };
                            let b_dim = if i < batch_rank - b_batch.len() {
                                1
                            } else {
                                b_batch[i - (batch_rank - b_batch.len())]
                            };
                            batch_shape[i] = a_dim.max(b_dim);
                        }

                        let batch_size: usize = batch_shape.iter().product();
                        let mut out_shape = batch_shape.clone();
                        out_shape.push(m);
                        out_shape.push(n);

                        let needs_alloc = match &*o {
                            TypedArray::Float(out) => out.shape() != out_shape.as_slice(),
                            _ => true,
                        };
                        if needs_alloc {
                            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&out_shape)))
                                .ensure_contiguous();
                        }

                        if let TypedArray::Float(out) = o {
                            let a_sl = a_arr.as_slice_memory_order().unwrap();
                            let b_sl = b_arr.as_slice_memory_order().unwrap();
                            let dst = out.as_slice_memory_order_mut().unwrap();

                            let a_mat_size = m * k;
                            let b_mat_size = k * n;
                            let o_mat_size = m * n;

                            let a_batch_size: usize = a_batch.iter().product::<usize>().max(1);
                            let b_batch_size: usize = b_batch.iter().product::<usize>().max(1);

                            for batch in 0..batch_size {
                                let a_batch_idx = batch % a_batch_size;
                                let b_batch_idx = batch % b_batch_size;

                                let a_offset = a_batch_idx * a_mat_size;
                                let b_offset = b_batch_idx * b_mat_size;
                                let o_offset = batch * o_mat_size;
                                sgemm_bias_parallel(
                                    m,
                                    n,
                                    k,
                                    &a_sl[a_offset..a_offset + a_mat_size],
                                    &b_sl[b_offset..b_offset + b_mat_size],
                                    None,
                                    &mut dst[o_offset..o_offset + o_mat_size],
                                    Activation::None,
                                );
                            }
                        }
                    }
                }

                Ok(())
            }
            _ => anyhow::bail!("MatMul: only F32 supported"),
        }
    }
}
