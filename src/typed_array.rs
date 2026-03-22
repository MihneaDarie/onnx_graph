use std::cell::RefCell;
use std::fmt::Display;
use std::mem;
use std::ops::BitAnd;

use crate::nodes::conv::Conv2D;
use crate::nodes::resize::Mode;
use crate::{
    argmax_variant, call_argmax_for_typed_array, call_concat_for_typed_array,
    call_gather_for_typed_array, call_maxpool_for_typed_array, call_pow_for_typed_array,
    call_reshape_for_typed_array, call_slice_for_typed_array, call_split_for_typed_array,
    call_transpose_for_typed_array, concat_variant, discriminant_macro, fix_if_not_contignous,
    from_shape_vec_from_datatype, gather_variant, get_curent_size_and_shape, impl_pow_variant,
    impl_typed_binop, impl_typed_singleopfunction, max_pool_variant, reshape_variant, shape_macro,
    slice_variant, softmax_variant, split_variant, transpose_variant, zeros_from_datatype,
    zeros_from_others_type,
};
use ndarray::{Array1, Array4, ArrayD, ArrayView1, Ix1, Ix4, IxDyn};
use ndarray::{ArrayView4, ArrayViewMut4, Axis};
use onnx_extractor::{DataType, OnnxTensor};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};
use saker_rs::linarg::operations::{apply_relu, sgemm_bias_parallel};
use saker_rs::{
    activations::Activation,
    linarg::operations::{apply_sigmoid, apply_silu},
};
use std::ops::Neg;

thread_local! {
    static POOL_TMP: RefCell<Vec<f32>> = const {RefCell::new(Vec::new())};
}

thread_local! {
    static IM2COL_BUF_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {std::cell::RefCell::new(Vec::new())};
}

pub fn maxpool_5x5(input: &ArrayView4<f32>, output: &mut ArrayViewMut4<f32>) {
    let (_, _, h, w) = input.dim();

    let in_sl = input.as_slice_memory_order().unwrap();
    let out_sl = output.as_slice_memory_order_mut().unwrap();

    let hw = h * w;

    out_sl
        .par_chunks_mut(hw)
        .enumerate()
        .for_each(|(ch, out_ch)| {
            let in_ch = &in_sl[ch * hw..(ch + 1) * hw];

            POOL_TMP.with(|cell| {
                let mut tmp = cell.borrow_mut();
                tmp.resize(hw, f32::NEG_INFINITY);

                for y in 0..h {
                    let row = y * w;
                    let tmp_row = &mut tmp[row..row + w];

                    for x in 0..w {
                        let x0 = x.saturating_sub(2);
                        let x1 = x.saturating_sub(1);
                        let x2 = x;
                        let x3 = (x + 1).min(w - 1);
                        let x4 = (x + 2).min(w - 1);

                        unsafe {
                            let a = *in_ch.get_unchecked(row + x0);
                            let b = *in_ch.get_unchecked(row + x1);
                            let c = *in_ch.get_unchecked(row + x2);
                            let d = *in_ch.get_unchecked(row + x3);
                            let e = *in_ch.get_unchecked(row + x4);
                            *tmp_row.get_unchecked_mut(x) = max5(&a, &b, &c, &d, &e);
                        }
                    }
                }

                for y in 0..h {
                    let y0 = y.saturating_sub(2);
                    let y1 = y.saturating_sub(1);
                    let y2 = y;
                    let y3 = (y + 1).min(h - 1);
                    let y4 = (y + 2).min(h - 1);

                    let r0 = y0 * w;
                    let r1 = y1 * w;
                    let r2 = y2 * w;
                    let r3 = y3 * w;
                    let r4 = y4 * w;

                    let out_row = &mut out_ch[y * w..y * w + w];

                    for x in 0..w {
                        unsafe {
                            let a = *tmp.get_unchecked(r0 + x);
                            let b = *tmp.get_unchecked(r1 + x);
                            let c0 = *tmp.get_unchecked(r2 + x);
                            let d = *tmp.get_unchecked(r3 + x);
                            let e = *tmp.get_unchecked(r4 + x);
                            *out_row.get_unchecked_mut(x) = max5(&a, &b, &c0, &d, &e);
                        }
                    }
                }
            });
        });
}

#[inline(always)]
fn max5(a: &f32, b: &f32, c: &f32, d: &f32, e: &f32) -> f32 {
    a.max(*b).max(*c).max(*d).max(*e)
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedArray {
    Undefined,
    Float(ArrayD<f32>),
    Uint8(ArrayD<u8>),
    Int8(ArrayD<i8>),
    Uint16(ArrayD<u16>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    String(ArrayD<String>),
    Bool(ArrayD<bool>),
    Double(ArrayD<f64>),
    Uint32(ArrayD<u32>),
    Uint64(ArrayD<u64>),
}

#[inline(always)]
pub fn aprox_silu_f32(x: f32) -> f32 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn aprox_silu_f64(x: f64) -> f64 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn relu_f64(x: f64) -> f64 {
    x.max(0.0f64)
}

#[inline(always)]
pub fn relu_f32(x: f64) -> f64 {
    x.max(0.0f64)
}

#[inline(always)]
pub fn aprox_sigmoid_f32(x: f32) -> f32 {
    aprox_silu_f32(x) / x
}

#[inline(always)]
pub fn aprox_sigmoid_f64(x: f64) -> f64 {
    aprox_silu_f64(x) / x
}

impl TypedArray {
    pub fn discriminatn(&self) -> String {
        discriminant_macro!(
            self,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, String, Bool, Double, Uint32,
                Uint64
            ]
        )
    }

    pub fn shape(&self) -> Option<&[usize]> {
        shape_macro!(
            self,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, String, Bool, Double, Uint32,
                Uint64
            ]
        )
    }

    pub fn ensure_contiguous(self) -> Self {
        fix_if_not_contignous!(
            self,
            [
                Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16, Bool,
                String
            ]
        )
    }

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

    pub fn shape_op(
        data: &TypedArray,
        start: i64,
        end: Option<i64>,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let shape: Vec<i64> = data
            .shape()
            .unwrap()
            .iter()
            .map(|val| *val as i64)
            .collect();

        let r = shape.len() as i64;

        let start = if start < 0 {
            (r + start).max(0) as usize
        } else {
            (start as usize).min(r as usize)
        };

        let end = match end {
            Some(e) => {
                if e < 0 {
                    (r + e).max(0) as usize
                } else {
                    (e as usize).min(r as usize)
                }
            }
            None => r as usize,
        };

        let sliced: Vec<i64> = if start >= end {
            vec![]
        } else {
            shape[start..end].to_vec()
        };

        let len = sliced.len();
        *o = TypedArray::Int64(ArrayD::from_shape_vec(IxDyn(&[len]), sliced).unwrap());

        Ok(())
    }

    pub fn max_pool(
        &self,
        kernel: &[usize],
        strides: &[usize],
        pads: &[usize],
        dilations: &[usize],
        ceil_mode: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        if let TypedArray::Float(x) = self {
            let kh = kernel[0];
            let kw = kernel[1];
            let sh = strides.first().copied().unwrap_or(1);
            let sw = strides.get(1).copied().unwrap_or(1);
            let ph = pads.first().copied().unwrap_or(0);
            let pw = pads.get(1).copied().unwrap_or(0);
            let dh = dilations.first().copied().unwrap_or(1);
            let dw = dilations.get(1).copied().unwrap_or(1);

            if kh == 5 && kw == 5 && sh == 1 && sw == 1 && ph == 2 && pw == 2 && dh == 1 && dw == 1
            {
                let x4 = x.view().into_dimensionality::<Ix4>()?;
                if let TypedArray::Float(o) = o {
                    let mut out4 = o.view_mut().into_dimensionality::<Ix4>()?;
                    maxpool_5x5(&x4, &mut out4);
                }
                return Ok(());
            }
        }

        call_maxpool_for_typed_array!(
            self,
            kernel,
            strides,
            pads,
            dilations,
            ceil_mode,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }

    pub fn slice(
        &self,
        starts: &TypedArray,
        ends: &TypedArray,
        axes: &TypedArray,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let starts = match starts {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("starts must be I64")),
        };
        let ends = match ends {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("ends must be I64")),
        };
        let axes = match axes {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("axes must be I64")),
        };

        call_slice_for_typed_array!(
            self,
            axes,
            starts,
            ends,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }

    pub fn split(
        &self,
        split: &TypedArray,
        axis: i64,
        outputs: &mut Vec<TypedArray>,
    ) -> anyhow::Result<()> {
        let splits = match split {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("split tensor must be I64")),
        };

        call_split_for_typed_array!(
            self,
            axis,
            splits,
            outputs,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }

    pub fn transpose(&self, perm: &[i64], o: &mut TypedArray) -> anyhow::Result<()> {
        call_transpose_for_typed_array!(
            self,
            perm,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );
        Ok(())
    }

    pub fn softmax(&self, axis: i64, o: &mut TypedArray) -> anyhow::Result<()> {
        match self {
            TypedArray::Float(a) => softmax_variant!(Float, axis, a, o, f32),
            TypedArray::Double(a) => softmax_variant!(Double, axis, a, o, f64),
            _ => return Err(anyhow::anyhow!("softmax only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn sigmoid(&self, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, &mut *o) {
            (TypedArray::Float(i), TypedArray::Float(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                apply_sigmoid(dst, src);
            }
            (TypedArray::Double(i), TypedArray::Double(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = aprox_sigmoid_f64(*s));
            }
            _ => return Err(anyhow::anyhow!("sigmoid only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn silu(&self, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, &mut *o) {
            (TypedArray::Float(i), TypedArray::Float(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                apply_silu(dst, src, i.len());
            }
            (TypedArray::Double(i), TypedArray::Double(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = aprox_silu_f64(*s));
            }
            _ => return Err(anyhow::anyhow!("silu only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn relu(&self, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, &mut *o) {
            (TypedArray::Float(i), TypedArray::Float(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                apply_relu(dst, src, i.len());
            }
            (TypedArray::Double(i), TypedArray::Double(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = relu_f64(*s));
            }
            _ => return Err(anyhow::anyhow!("relu only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn reshape(
        &self,
        shape: &TypedArray,
        allow_zero: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let shape_arr = match shape {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("reshape shape tensor must be I64")),
        };

        let (current_size, current_shape) = get_curent_size_and_shape!(
            self,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64, Bool
            ]
        );

        let mut new_shape: Vec<usize> = shape_arr
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if dim == -1 {
                    0
                } else if dim == 0 {
                    if allow_zero {
                        0
                    } else {
                        *current_shape.get(i).unwrap_or(&0)
                    }
                } else {
                    dim as usize
                }
            })
            .collect();

        if let Some(idx) = shape_arr.iter().position(|&d| d == -1) {
            let known: usize = new_shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != idx)
                .map(|(_, &d)| if d == 0 { 1 } else { d })
                .product();
            new_shape[idx] = current_size / known;
        }

        call_reshape_for_typed_array!(
            self,
            new_shape,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }

    pub fn concat(arrays: &[&TypedArray], axis: usize, o: &mut TypedArray) -> anyhow::Result<()> {
        call_concat_for_typed_array!(
            arrays[0],
            arrays,
            axis,
            o,
            [
                Float, Double, Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64
            ]
        );
        Ok(())
    }

    #[inline(always)]
    pub fn run_func_with_f32_buffer<R>(buf_size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
        IM2COL_BUF_POOL.with(|cell| {
            let mut buf = cell.borrow_mut().pop().unwrap_or_default();

            if buf.len() < buf_size {
                buf.resize(buf_size, 0.0f32);
            }

            let a = f(&mut buf[..buf_size]);

            cell.borrow_mut().push(buf);

            a
        })
    }

    #[inline]
    fn im2col_3x3_s1p1(input: &[f32], h: usize, w: usize, col_buffer: &mut [f32]) {
        let hw = h * w;
        col_buffer
            .par_chunks_mut(9 * hw)
            .enumerate()
            .for_each(|(ic, chunk)| {
                let in_c_base = ic * hw;

                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let k_idx = ky * 3 + kx;
                        let col_row = &mut chunk[k_idx * hw..(k_idx + 1) * hw];
                        let dy = ky as isize - 1;
                        let dx = kx as isize - 1;

                        for oy in 0..h {
                            let iy = oy as isize + dy;
                            let out_row_start = oy * w;

                            if iy < 0 || iy >= h as isize {
                                for ox in 0..w {
                                    col_row[out_row_start + ox] = 0.0;
                                }
                            } else {
                                let in_row_base = in_c_base + (iy as usize) * w;
                                for ox in 0..w {
                                    let ix = ox as isize + dx;
                                    col_row[out_row_start + ox] = if ix < 0 || ix >= w as isize {
                                        0.0
                                    } else {
                                        unsafe { *input.get_unchecked(in_row_base + ix as usize) }
                                    };
                                }
                            }
                        }
                    }
                }
            });
    }

    #[inline]
    fn im2col_3x3_s2p1(
        input: &[f32],
        hin: usize,
        win: usize,
        hout: usize,
        wout: usize,
        col_buffer: &mut [f32],
    ) {
        let hw_out = hout * wout;

        col_buffer
            .par_chunks_mut(9 * hw_out)
            .enumerate()
            .for_each(|(ic, chunk)| {
                let in_c_base = ic * hin * win;

                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let k_idx = ky * 3 + kx;
                        let col_row = &mut chunk[k_idx * hw_out..(k_idx + 1) * hw_out];

                        for oy in 0..hout {
                            let iy = (oy * 2 + ky) as isize - 1;
                            let out_row_start = oy * wout;

                            if iy < 0 || iy >= hin as isize {
                                for ox in 0..wout {
                                    col_row[out_row_start + ox] = 0.0;
                                }
                            } else {
                                let in_row_base = in_c_base + (iy as usize) * win;
                                for ox in 0..wout {
                                    let ix = (ox * 2 + kx) as isize - 1;
                                    col_row[out_row_start + ox] = if ix < 0 || ix >= win as isize {
                                        0.0
                                    } else {
                                        unsafe { *input.get_unchecked(in_row_base + ix as usize) }
                                    };
                                }
                            }
                        }
                    }
                }
            });
    }

    fn im2col_general(
        input: &[f32],
        cin: usize,
        hin: usize,
        win: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        hout: usize,
        wout: usize,
        col_buffer: &mut [f32],
    ) {
        let ksize = kh * kw;
        let hw_out = hout * wout;

        col_buffer
            .par_chunks_mut(ksize * hw_out)
            .enumerate()
            .for_each(|(ic, chunk)| {
                let in_c_base = ic * hin * win;

                for ky in 0..kh {
                    for kx in 0..kw {
                        let k_idx = ky * kw + kx;
                        let col_row = &mut chunk[k_idx * hw_out..(k_idx + 1) * hw_out];

                        for oy in 0..hout {
                            let iy = (oy * sh + ky) as isize - ph as isize;
                            let out_row_start = oy * wout;

                            if iy < 0 || iy >= hin as isize {
                                for ox in 0..wout {
                                    col_row[out_row_start + ox] = 0.0;
                                }
                            } else {
                                let in_row_base = in_c_base + (iy as usize) * win;
                                for ox in 0..wout {
                                    let ix = (ox * sw + kx) as isize - pw as isize;
                                    col_row[out_row_start + ox] = if ix < 0 || ix >= win as isize {
                                        0.0
                                    } else {
                                        unsafe { *input.get_unchecked(in_row_base + ix as usize) }
                                    };
                                }
                            }
                        }
                    }
                }
            });
    }

    pub fn conv_silu_into(
        x: &ArrayView4<f32>,
        w: &ArrayView4<f32>,
        conv_bias: Option<ArrayView1<f32>>,
        cfg: &Conv2D,
        out: &mut ArrayViewMut4<f32>,
        activation: Activation,
    ) -> anyhow::Result<()> {
        let (_, cin, hin, win) = x.dim();
        let (cout, _, kh, kw) = w.dim();

        if kh == 1 && kw == 1 && cfg.pad == 0 && cfg.stride == 1 {
            let hw = hin * win;
            let xs = x.as_slice_memory_order().unwrap();
            let ws = w.as_slice_memory_order().unwrap();
            let out_sl = out.as_slice_memory_order_mut().unwrap();
            let bias = conv_bias.as_ref().map(|b| b.as_slice().unwrap());

            sgemm_bias_parallel(cout, hw, cin, ws, xs, bias, out_sl, activation);
            return Ok(());
        }
        let hw = hin * win;

        let hout = (hin + 2 * cfg.pad - kh) / cfg.stride + 1;
        let wout = (win + 2 * cfg.pad - kw) / cfg.stride + 1;
        let hw_out = hout * wout;
        let k_dim = cin * kh * kw;

        let xs = x.as_slice_memory_order().unwrap();
        let ws = w.as_slice_memory_order().unwrap();
        let out_sl = out.as_slice_memory_order_mut().unwrap();

        let col_size = k_dim * hw_out;
        Self::run_func_with_f32_buffer(col_size, |col_buffer| {
            if kh == 3 && kw == 3 && cfg.stride == 1 && cfg.pad == 1 {
                Self::im2col_3x3_s1p1(xs, hin, win, col_buffer);
            } else if kh == 3 && kw == 3 && cfg.stride == 2 && cfg.pad == 1 {
                Self::im2col_3x3_s2p1(xs, hin, win, hout, wout, col_buffer);
            } else {
                Self::im2col_general(
                    xs, cin, hin, win, kh, kw, cfg.stride, cfg.stride, cfg.pad, cfg.pad, hout,
                    wout, col_buffer,
                );
            }

            let bias = conv_bias.as_ref().map(|b| b.as_slice().unwrap());
            sgemm_bias_parallel(
                cout, hw_out, k_dim, ws, col_buffer, bias, out_sl, activation,
            );
        });

        Ok(())
    }

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
            *o = TypedArray::Float(ArrayD::zeros(IxDyn(&expected)));
        }

        let out_arr = match o {
            TypedArray::Float(arr) => arr,
            _ => unreachable!(),
        };
        let out_sl = out_arr.as_slice_memory_order_mut().unwrap();
        let a_sl = a_arr.as_slice_memory_order().unwrap();
        let b_sl = b_arr.as_slice_memory_order().unwrap();

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
            let c_sl = c_arr.as_slice_memory_order().unwrap();
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

    pub fn conv(
        &self,
        w: &TypedArray,
        bias: Option<&TypedArray>,
        cfg: &Conv2D,
        o: &mut TypedArray,
        activation: Activation,
    ) -> anyhow::Result<()> {
        match (self, w, o) {
            (TypedArray::Float(x), TypedArray::Float(w), TypedArray::Float(o)) => {
                let x4 = x.view().into_dimensionality::<Ix4>()?;
                let w4 = w.view().into_dimensionality::<Ix4>()?;
                let mut out = o.view_mut().into_dimensionality::<Ix4>()?;

                let bias = bias
                    .map(|b| match b {
                        TypedArray::Float(b) => Ok(b.view().into_dimensionality::<Ix1>()?),
                        _ => Err(anyhow::anyhow!("bias must be F32")),
                    })
                    .transpose()?;

                Self::conv_silu_into(&x4, &w4, bias, cfg, &mut out, activation)?;
                Ok(())
            }
            (TypedArray::Undefined, _, _)
            | (_, TypedArray::Undefined, _)
            | (_, _, TypedArray::Undefined) => Err(anyhow::anyhow!("undefined type in conv")),
            _ => Err(anyhow::anyhow!("unsupported or mismatched types for conv")),
        }
    }

    pub fn resize(
        &self,
        sizes: Option<&TypedArray>,
        scales: Option<&TypedArray>,
        mode: &Mode,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        match self {
            TypedArray::Float(x) => {
                let x4 = x.view().into_dimensionality::<Ix4>()?;
                let (_, _, hin, win) = x4.dim();
                let in_sl = x4.as_slice_memory_order().unwrap();

                let (hout, wout) = match (sizes, scales) {
                    (Some(TypedArray::Int64(s)), _) => {
                        (s[s.len() - 2] as usize, s[s.len() - 1] as usize)
                    }
                    (_, Some(TypedArray::Float(s))) => {
                        let sh = s[s.len() - 2];
                        let sw = s[s.len() - 1];
                        ((hin as f32 * sh) as usize, (win as f32 * sw) as usize)
                    }
                    _ => return Err(anyhow::anyhow!("resize requires either sizes or scales")),
                };

                let out = match o {
                    TypedArray::Float(arr) => arr,
                    _ => unreachable!(),
                };
                let out_sl = out.as_slice_memory_order_mut().unwrap();

                let hw_in = hin * win;
                let hw_out = hout * wout;

                match mode {
                    Mode::Nearest => {
                        let rh = hin as f32 / hout as f32;
                        let rw = win as f32 / wout as f32;

                        let map_h: Vec<usize> = (0..hout)
                            .map(|oh| ((oh as f32 * rh) as usize).min(hin - 1))
                            .collect();
                        let map_w: Vec<usize> = (0..wout)
                            .map(|ow| ((ow as f32 * rw) as usize).min(win - 1))
                            .collect();

                        out_sl
                            .par_chunks_mut(hw_out)
                            .enumerate()
                            .for_each(|(ch, out_ch)| {
                                let in_ch = &in_sl[ch * hw_in..ch * hw_in + hw_in];
                                for oh in 0..hout {
                                    let ih = map_h[oh];
                                    let out_row = &mut out_ch[oh * wout..(oh + 1) * wout];
                                    let in_row_off = ih * win;
                                    for (ow, val) in map_w.iter().enumerate().take(wout) {
                                        unsafe {
                                            *out_row.get_unchecked_mut(ow) =
                                                *in_ch.get_unchecked(in_row_off + val);
                                        }
                                    }
                                }
                            });
                    }
                    Mode::Linear => {
                        let rh_scale = (hin as f32 - 1.0) / (hout as f32 - 1.0).max(1.0);
                        let rw_scale = (win as f32 - 1.0) / (wout as f32 - 1.0).max(1.0);

                        let h_params: Vec<(usize, usize, f32)> = (0..hout)
                            .map(|oh| {
                                let ih = oh as f32 * rh_scale;
                                let ih0 = (ih as usize).min(hin - 1);
                                let ih1 = (ih0 + 1).min(hin - 1);
                                (ih0, ih1, ih - ih0 as f32)
                            })
                            .collect();
                        let w_params: Vec<(usize, usize, f32)> = (0..wout)
                            .map(|ow| {
                                let iw = ow as f32 * rw_scale;
                                let iw0 = (iw as usize).min(win - 1);
                                let iw1 = (iw0 + 1).min(win - 1);
                                (iw0, iw1, iw - iw0 as f32)
                            })
                            .collect();

                        out_sl
                            .par_chunks_mut(hw_out)
                            .enumerate()
                            .for_each(|(ch, out_ch)| {
                                let in_ch = &in_sl[ch * hw_in..ch * hw_in + hw_in];
                                for oh in 0..hout {
                                    let (ih0, ih1, dh) = h_params[oh];
                                    let out_row = &mut out_ch[oh * wout..(oh + 1) * wout];
                                    let r0 = ih0 * win;
                                    let r1 = ih1 * win;
                                    for (ow, (iw0, iw1, dw)) in
                                        w_params.iter().enumerate().take(wout)
                                    {
                                        unsafe {
                                            let v00 = *in_ch.get_unchecked(r0 + iw0);
                                            let v01 = *in_ch.get_unchecked(r0 + iw1);
                                            let v10 = *in_ch.get_unchecked(r1 + iw0);
                                            let v11 = *in_ch.get_unchecked(r1 + iw1);
                                            *out_row.get_unchecked_mut(ow) =
                                                v00 * (1.0 - dh) * (1.0 - dw)
                                                    + v01 * (1.0 - dh) * dw
                                                    + v10 * dh * (1.0 - dw)
                                                    + v11 * dh * dw;
                                        }
                                    }
                                }
                            });
                    }
                    Mode::Cubic => {
                        return Err(anyhow::anyhow!("cubic resize not yet implemented"));
                    }
                }

                Ok(())
            }
            _ => Err(anyhow::anyhow!("resize only supported for F32")),
        }
    }

    impl_typed_binop!(add, +, +=, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
    impl_typed_binop!(sub, -, -=, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
    impl_typed_binop!(mul, *, *=, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);
    impl_typed_binop!(div, /, /=, [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64, Int8, Int16]);

    impl_typed_singleopfunction!(
        neg,
        neg,
        [Float, Double, Int32, Int64, Int8, Int16],
        [Uint8, Uint16, Uint32, Uint64]
    );

    pub fn and(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, b) {
            (TypedArray::Bool(a_arr), TypedArray::Bool(b_arr)) => {
                let needs_alloc = match &*o {
                    TypedArray::Bool(out) => out.shape() != a_arr.shape(),
                    _ => true,
                };
                if needs_alloc {
                    *o = TypedArray::Bool(ArrayD::default(IxDyn(a_arr.shape())));
                }
                if let TypedArray::Bool(o_arr) = o {
                    let dst = o_arr.as_slice_memory_order_mut().unwrap();
                    let a = a_arr.as_slice_memory_order().unwrap();
                    let b = b_arr.as_slice_memory_order().unwrap();
                    dst.iter_mut()
                        .zip(a.iter().zip(b.iter()))
                        .for_each(|(d, (a, b))| *d = *a & *b);
                }
            }
            _ => anyhow::bail!("Bitwise and only available for boolean tensors"),
        }
        Ok(())
    }

    pub fn pow(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
        let in_shape = self.shape().unwrap();
        call_pow_for_typed_array!(
            self,
            b,
            o,
            in_shape,
            [(Float, f32), (Double, f64), (Int32, i32), (Int64, i64)]
        );
        Ok(())
    }

    pub fn empty_with_others_type(other: &Self, shape: &[usize]) -> Self {
        zeros_from_others_type!(
            other,
            shape,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
            ]
        )
    }

    pub fn from_tensor_empty(tensor: &OnnxTensor, shape: &[i64]) -> Self {
        let shape = IxDyn(&shape.iter().map(|&x| x as usize).collect::<Vec<usize>>());

        zeros_from_datatype!(
            tensor.data_type(),
            shape,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
            ]
        )
    }

    pub fn from_output_tensor(tensor: &&OnnxTensor) -> Self {
        let shape = IxDyn(
            &tensor
                .shape()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
        );
        zeros_from_datatype!(
            tensor.data_type(),
            shape,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
            ]
        )
    }

    pub fn from_tensor(tensor: &&OnnxTensor) -> Self {
        let data_binding = tensor.data().unwrap();
        let binding = data_binding.as_slice();
        let data = binding.as_ref();
        let shape = IxDyn(
            &tensor
                .shape()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
        );

        from_shape_vec_from_datatype!(
            tensor.data_type(),
            shape,
            data,
            [
                (Float, f32),
                (Double, f64),
                (Int8, i8),
                (Int16, i16),
                (Int32, i32),
                (Int64, i64),
                (Uint8, u8),
                (Uint16, u16),
                (Uint32, u32),
                (Uint64, u64)
            ]
        )
    }
}

impl Display for TypedArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.shape())
    }
}
