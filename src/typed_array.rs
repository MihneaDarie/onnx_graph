use std::cell::RefCell;
use std::fmt::Display;

use crate::nodes::conv::Conv2D;
use crate::{
    discriminant_macro, fix_if_not_contignous, from_shape_vec_from_datatype, len_macro,
    shape_macro, zeros_from_datatype, zeros_from_others_type,
};
use anyhow::Ok;
use ndarray::{ArrayD, ArrayView1, IxDyn};
use ndarray::{ArrayView4, ArrayViewMut4};
use onnx_extractor::{DataType, OnnxTensor};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};
use saker_rs::activations::Activation;
use saker_rs::linarg::operations::sgemm_bias_parallel;

thread_local! {
    static POOL_TMP: RefCell<Vec<f32>> = const {RefCell::new(Vec::new())};
}

thread_local! {
    static IM2COL_BUF_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {std::cell::RefCell::new(Vec::new())};
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

    pub fn len(&self) -> Option<usize> {
        len_macro!(
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

    pub fn empty_with_others_type(other: &Self, shape: &[usize]) -> Self {
        zeros_from_others_type!(
            other,
            shape,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
            ]
        )
        .ensure_contiguous()
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
        .ensure_contiguous()
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
        .ensure_contiguous()
    }

    pub fn empty_from_data_type(data_type: DataType, shape: &[usize]) -> Self {
        zeros_from_datatype!(
            data_type,
            shape,
            [
                Float, Uint8, Int8, Uint16, Int16, Int32, Int64, Double, Uint32, Uint64
            ]
        )
        .ensure_contiguous()
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
        .ensure_contiguous()
    }
}

impl Display for TypedArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.shape())
    }
}
