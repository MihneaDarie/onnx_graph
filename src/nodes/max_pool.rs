use std::{any::Any, cell::RefCell, collections::HashMap, str::FromStr};

use crate::{
    call_maxpool_for_typed_array,
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use ndarray::Ix4;
use ndarray::{ArrayView4, ArrayViewMut4};
use onnx_extractor::{AttributeValue, OnnxOperation};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoPad {
    #[default]
    NOTSET,
    SameUpper,
    SameLower,
    VALID,
}

impl FromStr for AutoPad {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "SAME_UPPER" => Self::SameUpper,
            "SAME_LOWER" => Self::SameLower,
            "VALID" => Self::VALID,
            _ => Self::NOTSET,
        })
    }
}

#[derive(Default)]
pub struct MaxPoolNode<T: Default> {
    x: String,

    o: String,

    unique_id: UniqueId,

    auto_pad: AutoPad,
    ceil_mode: i64,
    kernel_shape: Vec<usize>,
    dilations: Vec<usize>,
    strides: Vec<usize>,
    pads: Vec<usize>,
    storage_order: usize,
    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for MaxPoolNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> anyhow::Result<Self> {
        let attrs = &elem.attributes;
        let mut max_pool = Self {
            x: String::new(),
            o: String::new(),
            auto_pad: {
                match attrs.get("auto_pad") {
                    Some(av) => {
                        let pad = av.as_string().unwrap();
                        AutoPad::from_str(pad).unwrap()
                    }
                    None => AutoPad::NOTSET,
                }
            },
            kernel_shape: {
                match attrs.get("kernel_shape") {
                    Some(av) => av
                        .as_ints()
                        .unwrap()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            pads: {
                match attrs.get("pads") {
                    Some(av) => av
                        .as_ints()
                        .unwrap()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            strides: {
                match attrs.get("strides") {
                    Some(av) => av
                        .as_ints()
                        .unwrap()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            dilations: {
                match attrs.get("dilations") {
                    Some(av) => av
                        .as_ints()
                        .unwrap()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            ceil_mode: {
                match attrs.get("ceil_mode") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            storage_order: {
                match attrs.get("storage_order") {
                    Some(av) => av.as_int().unwrap().to_owned() as usize,
                    None => 0,
                }
            },
            unique_id: UniqueId::MaxPool,
            next_node: None,
        };
        max_pool.add_input_strings(elem.inputs[0].clone());
        max_pool.add_output_strings(elem.outputs[0].clone());

        Ok(max_pool)
    }
}

impl<T: Default> MaxPoolNode<T> {
    pub fn new(
        auto_pad: &str,
        ceil_mode: i64,
        kernel_shape: Vec<usize>,
        dilations: Vec<usize>,
        strides: Vec<usize>,
        storage_order: usize,
        pads: Vec<usize>,
    ) -> Self {
        Self {
            x: String::new(),
            o: String::new(),
            auto_pad: AutoPad::from_str(auto_pad).unwrap(),
            ceil_mode,
            kernel_shape,
            dilations,
            strides,
            pads,
            storage_order,
            unique_id: UniqueId::MaxPool,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, x: String) {
        self.x = x;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

pub fn maxpool_fast(
    input: &TypedArray,
    kernel: &[usize],
    strides: &[usize],
    pads: &[usize],
    dilations: &[usize],
    o: &mut TypedArray,
) -> anyhow::Result<bool> {
    let kh = kernel[0];
    let kw = kernel[1];
    let sh = strides.first().copied().unwrap_or(1);
    let sw = strides.get(1).copied().unwrap_or(1);
    let ph = pads.first().copied().unwrap_or(0);
    let pw = pads.get(1).copied().unwrap_or(0);
    let dh = dilations.first().copied().unwrap_or(1);
    let dw = dilations.get(1).copied().unwrap_or(1);

    if let (TypedArray::Float(x), TypedArray::Float(out)) = (input, &mut *o)
        && kh == kw
        && sh == 1
        && sw == 1
        && ph == kh / 2
        && pw == kw / 2
        && dh == 1
        && dw == 1
    {
        let x4 = x.view().into_dimensionality::<Ix4>()?;

        let mut out4 = out.view_mut().into_dimensionality::<Ix4>()?;

        match kh {
            3 => maxpool_3x3_mut(&x4, &mut out4),
            5 => maxpool_5x5_mut(&x4, &mut out4),
            9 => maxpool_9x9_mut(&x4, &mut out4),
            13 => maxpool_13x13_mut(&x4, &mut out4),
            _ => return Ok(false),
        }

        return Ok(true);
    }

    Ok(false)
}

thread_local! {
    static POOL_TMP: RefCell<Vec<f32>> = const {RefCell::new(Vec::new())};
}

macro_rules! impl_maxpool_nxn {
    ($name:ident, $k:expr) => {
        pub fn $name(input: &ArrayView4<f32>, output: &mut ArrayViewMut4<f32>) {
            const K: usize = $k;
            const HALF: usize = K / 2;

            let (_, _, h, w) = input.dim();
            let hw = h * w;

            let in_sl = input.as_slice_memory_order().unwrap();
            let out_sl = output.as_slice_memory_order_mut().unwrap();

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
                                let x0 = x.saturating_sub(HALF);
                                let x1 = (x + HALF).min(w - 1);
                                let mut val = f32::NEG_INFINITY;
                                for xi in x0..=x1 {
                                    unsafe {
                                        let v = *in_ch.get_unchecked(row + xi);
                                        if v > val {
                                            val = v;
                                        }
                                    }
                                }
                                unsafe {
                                    *tmp_row.get_unchecked_mut(x) = val;
                                }
                            }
                        }

                        for y in 0..h {
                            let y0 = y.saturating_sub(HALF);
                            let y1 = (y + HALF).min(h - 1);
                            let out_row = &mut out_ch[y * w..y * w + w];
                            for x in 0..w {
                                let mut val = f32::NEG_INFINITY;
                                for yi in y0..=y1 {
                                    unsafe {
                                        let v = *tmp.get_unchecked(yi * w + x);
                                        if v > val {
                                            val = v;
                                        }
                                    }
                                }
                                unsafe {
                                    *out_row.get_unchecked_mut(x) = val;
                                }
                            }
                        }
                    });
                });
        }
    };
}

impl_maxpool_nxn!(maxpool_3x3_mut, 3);
impl_maxpool_nxn!(maxpool_5x5_mut, 5);
impl_maxpool_nxn!(maxpool_9x9_mut, 9);
impl_maxpool_nxn!(maxpool_13x13_mut, 13);

impl<T: Default + 'static> Node<T> for MaxPoolNode<T> {
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

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn input_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = &*x.unwrap();

        match o {
            Some(result) => {
                let kernel: Vec<usize> = self.kernel_shape.to_vec();
                let strides: Vec<usize> = self.strides.to_vec();
                let pads: Vec<usize> = self.pads.to_vec();
                let dilations: Vec<usize> = self.dilations.to_vec();

                let handled =
                    maxpool_fast(x, &kernel, &strides, &pads, &dilations, result).unwrap_or(false);

                if !handled {
                    x.max_pool(
                        &kernel,
                        &strides,
                        &pads,
                        &dilations,
                        self.ceil_mode != 0,
                        result,
                    )
                    .unwrap();
                }
            }
            None => panic!("MaxPoolNode: missing input {}", self.x),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("maxpool-{},{}", self.x, self.o);

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

    

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(x_shape) = x.shape()
        {
            let hin = x_shape[2];
            let win = x_shape[3];
            let kh = self.kernel_shape[0];
            let kw = self.kernel_shape[1];
            let sh = self.strides.first().copied().unwrap_or(1);
            let sw = self.strides.get(1).copied().unwrap_or(sh);
            let ph = self.pads.first().copied().unwrap_or(0);
            let pw = self.pads.get(1).copied().unwrap_or(ph);
            let dh = self.dilations.first().copied().unwrap_or(1);
            let dw = self.dilations.get(1).copied().unwrap_or(dh);

            let hout = if self.ceil_mode != 0 {
                (hin + 2 * ph - dh * (kh - 1) - 1 + sh - 1) / sh + 1
            } else {
                (hin + 2 * ph - dh * (kh - 1) - 1) / sh + 1
            };
            let wout = if self.ceil_mode != 0 {
                (win + 2 * pw - dw * (kw - 1) - 1 + sw - 1) / sw + 1
            } else {
                (win + 2 * pw - dw * (kw - 1) - 1) / sw + 1
            };

            let out_shape = &[x_shape[0], x_shape[1], hout, wout];
            *o = TypedArray::empty_with_others_type(x, out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

fn maxpool_5x5(input: &ArrayView4<f32>, output: &mut ArrayViewMut4<f32>) {
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

impl TypedArray {
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
}
