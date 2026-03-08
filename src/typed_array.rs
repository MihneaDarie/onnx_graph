use std::cell::RefCell;

use crate::nodes::conv::Conv2D;
use crate::nodes::resize::Mode;
use ndarray::{Array1, Array4, ArrayD, ArrayView1, Ix1, Ix4, IxDyn};
use ndarray::{ArrayView4, ArrayViewMut4, Axis};
use onnx_extractor::{DataType, OnnxTensor};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};
use saker_rs::linarg::operations::sgemm_bias_parallel;
use saker_rs::{
    activations::Activation,
    linarg::operations::{apply_sigmoid, apply_silu},
};

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
    F32(ArrayD<f32>),
    U8(ArrayD<u8>),
    I8(ArrayD<i8>),
    U16(ArrayD<u16>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    String(ArrayD<String>),
    BOOL(ArrayD<bool>),
    F64(ArrayD<f64>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
}

macro_rules! impl_typed_binop {
    ($name:ident, $op:tt, $op_assign:tt, [$($variant:ident),+]) => {
        pub fn $name(&self, b: &TypedArray, o: &mut TypedArray) -> anyhow::Result<()> {
            match (self, b) {
                $(
                    (TypedArray::$variant(a), TypedArray::$variant(b)) => {
                        if a.shape() == b.shape() {
                            if let TypedArray::$variant(out) = o {
                                    let dst = out.as_slice_memory_order_mut().unwrap();
                                    let sa = a.as_slice_memory_order().unwrap();
                                    let sb = b.as_slice_memory_order().unwrap();
                                    dst.par_iter_mut()
                                        .zip(sa.par_iter().zip(sb.par_iter()))
                                        .for_each(|(d, (a, b))| *d = *a $op *b);
                            } else {
                                *o = TypedArray::$variant(a $op b);
                            };

                        } else {
                            *o = TypedArray::$variant(a $op b);
                        }
                    }
                )+
                (TypedArray::Undefined, _) | (_, TypedArray::Undefined) => {
                    return Err(anyhow::anyhow!("undefined type: {}", stringify!($name)));
                }
                _ => return Err(anyhow::anyhow!("mismatch types {}", stringify!($name))),
            }
            Ok(())
        }
    };
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
pub fn aprox_sigmoid_f32(x: f32) -> f32 {
    aprox_silu_f32(x) / x
}

#[inline(always)]
pub fn aprox_sigmoid_f64(x: f64) -> f64 {
    aprox_silu_f64(x) / x
}

impl TypedArray {
    pub fn ensure_contiguous(self) -> Self {
        macro_rules! fix {
            ($variant:ident, $a:expr) => {
                if $a.is_standard_layout() {
                    TypedArray::$variant($a)
                } else {
                    TypedArray::$variant($a.as_standard_layout().into_owned())
                }
            };
        }
        match self {
            TypedArray::F32(a) => fix!(F32, a),
            TypedArray::F64(a) => fix!(F64, a),
            TypedArray::I32(a) => fix!(I32, a),
            TypedArray::I64(a) => fix!(I64, a),
            TypedArray::U8(a) => fix!(U8, a),
            TypedArray::U16(a) => fix!(U16, a),
            TypedArray::U32(a) => fix!(U32, a),
            TypedArray::U64(a) => fix!(U64, a),
            other => other,
        }
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
        if let TypedArray::F32(x) = self {
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
                if let TypedArray::F32(o) = o {
                    let mut out4 = o.view_mut().into_dimensionality::<Ix4>()?;
                    maxpool_5x5(&x4, &mut out4);
                }
                return Ok(());
            }
        }
        macro_rules! max_pool_variant {
            ($variant:ident, $x:expr) => {{
                let x4 = $x.view().into_dimensionality::<Ix4>()?;
                let (batch, channels, hin, win) = x4.dim();

                let kh = kernel[0];
                let kw = kernel[1];
                let sh = strides.first().copied().unwrap_or(1);
                let sw = strides.get(1).copied().unwrap_or(1);
                let ph = pads.first().copied().unwrap_or(0);
                let pw = pads.get(1).copied().unwrap_or(0);
                let dh = dilations.first().copied().unwrap_or(1);
                let dw = dilations.get(1).copied().unwrap_or(1);

                let hout = if ceil_mode {
                    (hin + 2 * ph - dh * (kh - 1) - 1).div_ceil(sh)
                } else {
                    (hin + 2 * ph - dh * (kh - 1) - 1) / sh + 1
                };
                let wout = if ceil_mode {
                    (win + 2 * pw - dw * (kw - 1) - 1).div_ceil(sw)
                } else {
                    (win + 2 * pw - dw * (kw - 1) - 1) / sw + 1
                };

                let mut out = ArrayD::from_elem(
                    IxDyn(&[batch, channels, hout, wout]),
                    $x.iter().next().copied().unwrap(),
                );

                for b in 0..batch {
                    for c in 0..channels {
                        for oh in 0..hout {
                            for ow in 0..wout {
                                let mut max_val = None;
                                for khi in 0..kh {
                                    for kwi in 0..kw {
                                        let ih = oh * sh + khi * dh;
                                        let iw = ow * sw + kwi * dw;
                                        if ih >= ph && iw >= pw {
                                            let ih = ih - ph;
                                            let iw = iw - pw;
                                            if ih < hin && iw < win {
                                                let val = x4[[b, c, ih, iw]];
                                                max_val = Some(match max_val {
                                                    None => val,
                                                    Some(m) => {
                                                        if val > m {
                                                            val
                                                        } else {
                                                            m
                                                        }
                                                    }
                                                });
                                            }
                                        }
                                    }
                                }
                                out[[b, c, oh, ow]] = max_val.unwrap();
                            }
                        }
                    }
                }
                *o = TypedArray::$variant(out);
            }};
        }

        match self {
            TypedArray::F32(x) => max_pool_variant!(F32, x),
            TypedArray::F64(x) => max_pool_variant!(F64, x),
            TypedArray::I32(x) => max_pool_variant!(I32, x),
            TypedArray::I64(x) => max_pool_variant!(I64, x),
            TypedArray::U8(x) => max_pool_variant!(U8, x),
            TypedArray::U16(x) => max_pool_variant!(U16, x),
            TypedArray::U32(x) => max_pool_variant!(U32, x),
            TypedArray::U64(x) => max_pool_variant!(U64, x),
            _ => return Err(anyhow::anyhow!("unsupported type for max_pool")),
        }
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
            TypedArray::I64(s) => s,
            _ => return Err(anyhow::anyhow!("starts must be I64")),
        };
        let ends = match ends {
            TypedArray::I64(s) => s,
            _ => return Err(anyhow::anyhow!("ends must be I64")),
        };
        let axes = match axes {
            TypedArray::I64(s) => s,
            _ => return Err(anyhow::anyhow!("axes must be I64")),
        };

        macro_rules! slice_variant {
            ($variant:ident, $a:expr) => {{
                let ndim = $a.ndim();
                let mut slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
                    .map(|_| ndarray::SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    })
                    .collect();

                for i in 0..axes.len() {
                    let axis = axes[i] as usize;
                    let dim_size = $a.shape()[axis] as i64;

                    let start = {
                        let s = starts[i];
                        if s < 0 {
                            (dim_size + s).max(0)
                        } else {
                            s.min(dim_size)
                        }
                    } as usize;

                    let end = {
                        let e = ends[i];
                        if e < 0 {
                            (dim_size + e).max(0)
                        } else {
                            e.min(dim_size)
                        }
                    } as usize;

                    slice_info[axis] = ndarray::SliceInfoElem::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    };
                }

                let view = $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?);

                if let TypedArray::$variant(out) = o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    for (d, s) in dst.iter_mut().zip(view.iter()) {
                        *d = *s;
                    }
                }
            }};
        }

        match self {
            TypedArray::F32(a) => slice_variant!(F32, a),
            TypedArray::F64(a) => slice_variant!(F64, a),
            TypedArray::I32(a) => slice_variant!(I32, a),
            TypedArray::I64(a) => slice_variant!(I64, a),
            TypedArray::U8(a) => slice_variant!(U8, a),
            TypedArray::U16(a) => slice_variant!(U16, a),
            TypedArray::U32(a) => slice_variant!(U32, a),
            TypedArray::U64(a) => slice_variant!(U64, a),
            _ => return Err(anyhow::anyhow!("unsupported type for slice")),
        }
        Ok(())
    }

    pub fn split(
        &self,
        split: &TypedArray,
        axis: i64,
        outputs: &mut Vec<TypedArray>,
    ) -> anyhow::Result<()> {
        let splits = match split {
            TypedArray::I64(s) => s,
            _ => return Err(anyhow::anyhow!("split tensor must be I64")),
        };

        macro_rules! split_variant {
            ($variant:ident, $a:expr) => {{
                let ndim = $a.ndim() as i64;
                let axis = if axis < 0 {
                    (ndim + axis) as usize
                } else {
                    axis as usize
                };

                let mut offset = 0;
                for &size in splits.iter() {
                    let size = size as usize;
                    let slice_info: Vec<ndarray::SliceInfoElem> = (0..$a.ndim())
                        .map(|i| {
                            if i == axis {
                                ndarray::SliceInfoElem::Slice {
                                    start: offset as isize,
                                    end: Some((offset + size) as isize),
                                    step: 1,
                                }
                            } else {
                                ndarray::SliceInfoElem::Slice {
                                    start: 0,
                                    end: None,
                                    step: 1,
                                }
                            }
                        })
                        .collect();

                    outputs.push(TypedArray::$variant(
                        $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?)
                            .to_owned(),
                    ));
                    offset += size;
                }
            }};
        }

        match self {
            TypedArray::F32(a) => split_variant!(F32, a),
            TypedArray::F64(a) => split_variant!(F64, a),
            TypedArray::I32(a) => split_variant!(I32, a),
            TypedArray::I64(a) => split_variant!(I64, a),
            TypedArray::U8(a) => split_variant!(U8, a),
            TypedArray::U16(a) => split_variant!(U16, a),
            TypedArray::U32(a) => split_variant!(U32, a),
            TypedArray::U64(a) => split_variant!(U64, a),
            _ => return Err(anyhow::anyhow!("unsupported type for split")),
        }
        Ok(())
    }

    pub fn transpose(&self, perm: &[i64], o: &mut TypedArray) -> anyhow::Result<()> {
        macro_rules! transpose_variant {
            ($variant:ident, $a:expr) => {{
                let ndim = $a.ndim();
                let perm: Vec<usize> = if perm.is_empty() {
                    (0..ndim).rev().collect()
                } else {
                    perm.iter()
                        .map(|&p| {
                            if p < 0 {
                                (ndim as i64 + p) as usize
                            } else {
                                p as usize
                            }
                        })
                        .collect()
                };

                let view = $a.view().permuted_axes(&*perm);

                if let TypedArray::$variant(out) = o {
                    ndarray::Zip::from(out).and(&view).par_for_each(|d, s| {
                        *d = *s;
                    });
                }
            }};
        }

        match self {
            TypedArray::F32(a) => transpose_variant!(F32, a),
            TypedArray::F64(a) => transpose_variant!(F64, a),
            TypedArray::I32(a) => transpose_variant!(I32, a),
            TypedArray::I64(a) => transpose_variant!(I64, a),
            TypedArray::U8(a) => transpose_variant!(U8, a),
            TypedArray::U16(a) => transpose_variant!(U16, a),
            TypedArray::U32(a) => transpose_variant!(U32, a),
            TypedArray::U64(a) => transpose_variant!(U64, a),
            _ => return Err(anyhow::anyhow!("unsupported type for transpose")),
        }
        Ok(())
    }

    pub fn softmax(&self, axis: i64, o: &mut TypedArray) -> anyhow::Result<()> {
        macro_rules! softmax_variant {
            ($variant:ident, $a:expr, $T:ty) => {{
                let ndim = $a.ndim() as i64;
                let axis = if axis < 0 {
                    (ndim + axis) as usize
                } else {
                    axis as usize
                };

                if let TypedArray::$variant(out) = o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    let src = $a.as_slice_memory_order().unwrap();
                    dst.copy_from_slice(src);

                    for mut lane in out.lanes_mut(Axis(axis)) {
                        let max = lane.iter().copied().fold(<$T>::NEG_INFINITY, <$T>::max);
                        lane.mapv_inplace(|x| (x - max).exp());
                        let sum: $T = lane.iter().sum();
                        lane.mapv_inplace(|x| x / sum);
                    }
                }
            }};
        }

        match self {
            TypedArray::F32(a) => softmax_variant!(F32, a, f32),
            TypedArray::F64(a) => softmax_variant!(F64, a, f64),
            _ => return Err(anyhow::anyhow!("softmax only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn sigmoid(&self, o: &mut TypedArray) -> anyhow::Result<()> {
        match (self, &mut *o) {
            (TypedArray::F32(i), TypedArray::F32(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                apply_sigmoid(dst, src);
            }
            (TypedArray::F64(i), TypedArray::F64(o)) => {
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
            (TypedArray::F32(i), TypedArray::F32(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                apply_silu(dst, src, i.len());
            }
            (TypedArray::F64(i), TypedArray::F64(o)) => {
                let src = i.as_slice_memory_order().unwrap();
                let dst = o.as_slice_memory_order_mut().unwrap();
                dst.par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, s)| *d = aprox_silu_f64(*s));
            }
            _ => return Err(anyhow::anyhow!("sigmoid only supported for F32/F64")),
        }
        Ok(())
    }

    pub fn reshape(
        &self,
        shape: &TypedArray,
        allow_zero: bool,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        macro_rules! reshape_variant {
            ($variant:ident, $a:expr) => {{
                if let TypedArray::$variant(out) = o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    let src = $a.as_slice_memory_order().unwrap();
                    dst.par_iter_mut()
                        .zip(src.par_iter())
                        .for_each(|(d, s)| *d = *s);
                }
            }};
        }

        match self {
            TypedArray::F32(a) => reshape_variant!(F32, a),
            TypedArray::F64(a) => reshape_variant!(F64, a),
            TypedArray::I32(a) => reshape_variant!(I32, a),
            TypedArray::I64(a) => reshape_variant!(I64, a),
            TypedArray::U8(a) => reshape_variant!(U8, a),
            TypedArray::U16(a) => reshape_variant!(U16, a),
            TypedArray::U32(a) => reshape_variant!(U32, a),
            TypedArray::U64(a) => reshape_variant!(U64, a),
            _ => return Err(anyhow::anyhow!("unsupported type for reshape")),
        }
        Ok(())
    }

    pub fn concat(arrays: &[&TypedArray], axis: usize, o: &mut TypedArray) -> anyhow::Result<()> {
        macro_rules! concat_variant {
            ($variant:ident, $arrays:expr, $axis:expr, $o:expr) => {{
                let inner: anyhow::Result<Vec<_>> = $arrays
                    .iter()
                    .map(|a| match a {
                        TypedArray::$variant(arr) => Ok(arr.view()),
                        _ => Err(anyhow::anyhow!("type mismatch in concat")),
                    })
                    .collect();
                *$o = TypedArray::$variant(ndarray::concatenate(Axis($axis), &inner?)?.into_dyn());
            }};
        }

        match arrays[0] {
            TypedArray::F32(_) => concat_variant!(F32, arrays, axis, o),
            TypedArray::F64(_) => concat_variant!(F64, arrays, axis, o),
            TypedArray::I32(_) => concat_variant!(I32, arrays, axis, o),
            TypedArray::I64(_) => concat_variant!(I64, arrays, axis, o),
            TypedArray::U8(_) => concat_variant!(U8, arrays, axis, o),
            TypedArray::U16(_) => concat_variant!(U16, arrays, axis, o),
            TypedArray::U32(_) => concat_variant!(U32, arrays, axis, o),
            TypedArray::U64(_) => concat_variant!(U64, arrays, axis, o),
            TypedArray::I8(_) => concat_variant!(I8, arrays, axis, o),
            TypedArray::I16(_) => concat_variant!(I16, arrays, axis, o),
            TypedArray::Undefined => return Err(anyhow::anyhow!("undefined type in concat")),
            _ => return Err(anyhow::anyhow!("unsupported type for concat")),
        }
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

        let hout = (hin + 2 * cfg.pad - kh) / cfg.stride + 1;
        let wout = (win + 2 * cfg.pad - kw) / cfg.stride + 1;
        let hw_out = hout * wout;

        let xs = x.as_slice_memory_order().unwrap();
        let ws = w.as_slice_memory_order().unwrap();
        let out_sl = out.as_slice_memory_order_mut().unwrap();

        let col_size = cin * 9 * hw_out;
        Self::run_func_with_f32_buffer(col_size, |col_buffer| {
            if cfg.stride == 1 && cfg.pad == 1 {
                Self::im2col_3x3_s1p1(xs, hin, win, col_buffer);
            } else if cfg.stride == 2 && cfg.pad == 1 {
                Self::im2col_3x3_s2p1(xs, hin, win, hout, wout, col_buffer);
            }

            let k_dim = cin * 9;
            let bias = conv_bias.as_ref().map(|b| b.as_slice().unwrap());

            sgemm_bias_parallel(
                cout, hw_out, k_dim, ws, col_buffer, bias, out_sl, activation,
            );
        });

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
            (TypedArray::F32(x), TypedArray::F32(w), TypedArray::F32(o)) => {
                let x4 = x.view().into_dimensionality::<Ix4>()?;
                let w4 = w.view().into_dimensionality::<Ix4>()?;
                let mut out = o.view_mut().into_dimensionality::<Ix4>()?;

                let bias = bias
                    .map(|b| match b {
                        TypedArray::F32(b) => Ok(b.view().into_dimensionality::<Ix1>()?),
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
            TypedArray::F32(x) => {
                let x4 = x.view().into_dimensionality::<Ix4>()?;
                let (_, _, hin, win) = x4.dim();
                let in_sl = x4.as_slice_memory_order().unwrap();

                let (hout, wout) = match (sizes, scales) {
                    (Some(TypedArray::I64(s)), _) => {
                        (s[s.len() - 2] as usize, s[s.len() - 1] as usize)
                    }
                    (_, Some(TypedArray::F32(s))) => {
                        let sh = s[s.len() - 2];
                        let sw = s[s.len() - 1];
                        ((hin as f32 * sh) as usize, (win as f32 * sw) as usize)
                    }
                    _ => return Err(anyhow::anyhow!("resize requires either sizes or scales")),
                };

                let out = match o {
                    TypedArray::F32(arr) => arr,
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

    impl_typed_binop!(add, +, +=, [F32, F64, I32, I64, U8, U16, U32, U64, I8, I16]);
    impl_typed_binop!(sub, -, -=, [F32, F64, I32, I64, U8, U16, U32, U64, I8, I16]);
    impl_typed_binop!(mul, *, *=, [F32, F64, I32, I64, U8, U16, U32, U64, I8, I16]);
    impl_typed_binop!(div, /, /=, [F32, F64, I32, I64, U8, U16, U32, U64, I8, I16]);

    pub fn from_tensor_empty(tensor: &OnnxTensor) -> Self {
        let shape = IxDyn(
            &tensor
                .shape()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
        );

        match tensor.data_type() {
            DataType::Float => TypedArray::F32(ArrayD::zeros(shape)),
            DataType::Double => TypedArray::F64(ArrayD::zeros(shape)),
            DataType::Int32 => TypedArray::I32(ArrayD::zeros(shape)),
            DataType::Int64 => TypedArray::I64(ArrayD::zeros(shape)),
            DataType::Uint8 => TypedArray::U8(ArrayD::zeros(shape)),
            DataType::Uint16 => TypedArray::U16(ArrayD::zeros(shape)),
            DataType::Uint32 => TypedArray::U32(ArrayD::zeros(shape)),
            DataType::Uint64 => TypedArray::U64(ArrayD::zeros(shape)),
            DataType::Int8 => TypedArray::I8(ArrayD::zeros(shape)),
            DataType::Int16 => TypedArray::I16(ArrayD::zeros(shape)),
            DataType::Bool => TypedArray::BOOL(ArrayD::from_elem(shape, false)),
            _ => TypedArray::Undefined,
        }
    }

    pub fn from_output_tensor(tensor: &&OnnxTensor) -> Self {
        let shape = IxDyn(
            &tensor
                .shape()
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<usize>>(),
        );
        match tensor.data_type() {
            DataType::Float => TypedArray::F32(ArrayD::zeros(shape)),
            DataType::Uint8 => TypedArray::U8(ArrayD::zeros(shape)),
            DataType::Int8 => TypedArray::I8(ArrayD::zeros(shape)),
            DataType::Uint16 => TypedArray::U16(ArrayD::zeros(shape)),
            DataType::Int16 => TypedArray::I16(ArrayD::zeros(shape)),
            DataType::Int32 => TypedArray::I32(ArrayD::zeros(shape)),
            DataType::Int64 => TypedArray::I64(ArrayD::zeros(shape)),
            DataType::Double => TypedArray::F64(ArrayD::zeros(shape)),
            DataType::Uint32 => TypedArray::U32(ArrayD::zeros(shape)),
            DataType::Uint64 => TypedArray::U64(ArrayD::zeros(shape)),
            _ => TypedArray::Undefined,
        }
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
        match tensor.data_type() {
            DataType::Float => {
                let floats = data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<f32>>();
                TypedArray::F32(ArrayD::from_shape_vec(shape, floats).unwrap())
            }
            DataType::Uint8 => {
                let u8s = data.to_vec();
                TypedArray::U8(ArrayD::from_shape_vec(shape, u8s).unwrap())
            }
            DataType::Int8 => {
                let i8s = data.iter().map(|&b| b as i8).collect::<Vec<i8>>();
                TypedArray::I8(ArrayD::from_shape_vec(shape, i8s).unwrap())
            }
            DataType::Uint16 => {
                let u16s = data
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<u16>>();
                TypedArray::U16(ArrayD::from_shape_vec(shape, u16s).unwrap())
            }
            DataType::Int16 => {
                let i16s = data
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<i16>>();
                TypedArray::I16(ArrayD::from_shape_vec(shape, i16s).unwrap())
            }
            DataType::Int32 => {
                let i32s = data
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<i32>>();
                TypedArray::I32(ArrayD::from_shape_vec(shape, i32s).unwrap())
            }
            DataType::Int64 => {
                let i64s = data
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<i64>>();
                TypedArray::I64(ArrayD::from_shape_vec(shape, i64s).unwrap())
            }
            DataType::Bool => {
                let bools = data.iter().map(|&b| b != 0).collect::<Vec<bool>>();
                TypedArray::BOOL(ArrayD::from_shape_vec(shape, bools).unwrap())
            }
            DataType::Double => {
                let f64s = data
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<f64>>();
                TypedArray::F64(ArrayD::from_shape_vec(shape, f64s).unwrap())
            }
            DataType::Uint32 => {
                let u32s = data
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<u32>>();
                TypedArray::U32(ArrayD::from_shape_vec(shape, u32s).unwrap())
            }
            DataType::Uint64 => {
                let u64s = data
                    .chunks_exact(8)
                    .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<u64>>();
                TypedArray::U64(ArrayD::from_shape_vec(shape, u64s).unwrap())
            }
            _ => TypedArray::Undefined,
        }
    }
}
