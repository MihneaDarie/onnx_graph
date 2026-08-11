use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

use anyhow::{Ok, Result};
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::{AttributeValue, OnnxOperation};
use saker_rs::{activations::Activation, linarg::operations::sgemm_bias_parallel};

#[derive(Default)]
pub struct GemmNode<T: Default> {
    a: u64,
    b: u64,
    c: Option<u64>,

    o: u64,

    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,

    unique_id: UniqueId,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for GemmNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes();
        let mut gemm = Self {
            a: u64::default(),
            b: u64::default(),
            c: None,
            o: u64::default(),
            alpha: attrs.get("alpha").and_then(|v| v.as_float()).unwrap_or(1.0),
            beta: attrs.get("beta").and_then(|v| v.as_float()).unwrap_or(1.0),
            trans_a: attrs.get("transA").and_then(|v| v.as_int()).unwrap_or(0) != 0,
            trans_b: attrs.get("transB").and_then(|v| v.as_int()).unwrap_or(0) != 0,
            unique_id: UniqueId::Gemm,
            next_node: None,
        };
        let inputs = &elem.inputs();
        let b = inputs.get(2).cloned().and_then(|val| {
            let id = hash_string(&val);
            Some(id)
        });

        let x_id = hash_string(&elem.inputs()[0]);
        let w_id = hash_string(&elem.inputs()[1]);
        gemm.add_inputs(x_id, w_id, b);
        let o_id = hash_string(&elem.outputs()[0]);
        gemm.add_outputs(o_id);
        Ok(gemm)
    }
}

impl<T: Default> GemmNode<T> {
    pub fn add_inputs(&mut self, a: u64, b: u64, c: Option<u64>) {
        self.a = a;
        self.b = b;
        self.c = c;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for GemmNode<T> {
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
        let mut names = vec![self.a.clone(), self.b.clone()];
        if let Some(c) = &self.c {
            names.push(*c);
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
        let def = u64::default();
        let c_key = self.c.as_ref().unwrap_or(&def);

        let [a, b, c, o] = omap.get_disjoint_mut([&self.a, &self.b, c_key, &self.o]);
        crate::debug_check_tensors!("GemmNode", a => self.a, b => self.b, o => self.o);
        if self.c.is_some() {
            crate::debug_check_tensors!("GemmNode", c => *c_key);
        }
        let a = a.map(|val| &*val);
        let b = b.map(|val| &*val);
        let c = if self.c.is_some() {
            c.map(|val| &*val)
        } else {
            None
        };
        if let (Some(a), Some(b), Some(out)) = (a, b, o) {
            TypedArray::gemm(
                a,
                b,
                c,
                self.alpha,
                self.beta,
                self.trans_a,
                self.trans_b,
                out,
            )?;
        }
        Ok(())
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "gemm-{},{},{:?},{}  alpha={} beta={} transA={} transB={}",
            self.a, self.b, self.c, self.o, self.alpha, self.beta, self.trans_a, self.trans_b
        );
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [a, b, o] = omap.get_disjoint_mut([&self.a, &self.b, &self.o]);
        let a = a.map(|arr| &*arr);
        let b = b.map(|arr| &*arr);

        if let (Some(a), Some(b), Some(o)) = (a, b, o)
            && let (Some(a_shape), Some(b_shape)) = (a.shape(), b.shape())
        {
            let m = if self.trans_a { a_shape[1] } else { a_shape[0] };
            let n = if self.trans_b { b_shape[0] } else { b_shape[1] };

            *o = TypedArray::empty_with_others_type(a, &[m, n]);
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
    pub fn gemm(
        a: &TypedArray,
        b: &TypedArray,
        c: Option<&TypedArray>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let a_arr = match a {
            TypedArray::Float(a) => a,
            _ => return Err(anyhow::anyhow!("Gemm: A must be F32")),
        };
        let b_arr = match b {
            TypedArray::Float(b) => b,
            _ => return Err(anyhow::anyhow!("Gemm: B must be F32")),
        };

        let a_shape = a_arr.shape();
        let b_shape = b_arr.shape();

        let (m, k) = if trans_a {
            (a_shape[1], a_shape[0])
        } else {
            (a_shape[0], a_shape[1])
        };
        let n = if trans_b { b_shape[0] } else { b_shape[1] };

        let expected = [m, n];
        let needs_alloc = match &*o {
            TypedArray::Float(out) => out.shape() != expected,
            _ => true,
        };
        if needs_alloc {
            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&expected))).ensure_contiguous();
        }

        let out_arr = match o {
            TypedArray::Float(arr) => arr,
            _ => unreachable!(),
        };
        let out_sl = slice_memory_order_mut_or_fix(out_arr, "gemm")?;
        let mut a_buf = a_arr.clone();
        let mut b_buf = b_arr.clone();
        let a_sl = slice_memory_order_or_fix(&mut a_buf, "gemm")?;
        let b_sl = slice_memory_order_or_fix(&mut b_buf, "gemm")?;

        let a_ready: Vec<f32>;
        let a_ptr = if trans_a {
            let rows = a_shape[0];
            let cols = a_shape[1];
            a_ready = (0..cols)
                .flat_map(|i| (0..rows).map(move |j| a_sl[j * cols + i]))
                .collect();
            &a_ready[..]
        } else {
            a_sl
        };

        let b_ready: Vec<f32>;
        let b_ptr = if trans_b {
            let rows = b_shape[0];
            let cols = b_shape[1];
            b_ready = (0..cols)
                .flat_map(|i| (0..rows).map(move |j| b_sl[j * cols + i]))
                .collect();
            &b_ready[..]
        } else {
            b_sl
        };

        sgemm_bias_parallel(m, n, k, a_ptr, b_ptr, None, out_sl, Activation::None);

        if alpha != 1.0 {
            out_sl.iter_mut().for_each(|v| *v *= alpha);
        }

        if let Some(TypedArray::Float(c_arr)) = c {
            let mut c_buf = c_arr.clone();
            let c_sl = slice_memory_order_or_fix(&mut c_buf, "gemm")?;
            if c_arr.len() == n {
                for row in 0..m {
                    let offset = row * n;
                    for col in 0..n {
                        out_sl[offset + col] += beta * c_sl[col];
                    }
                }
            } else if c_arr.len() == 1 {
                let val = beta * c_sl[0];
                out_sl.iter_mut().for_each(|v| *v += val);
            } else if c_arr.len() == m * n {
                for i in 0..m * n {
                    out_sl[i] += beta * c_sl[i];
                }
            }
        }

        Ok(())
    }
}
