use std::{any::Any, collections::HashMap, str::FromStr};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    nodes_utils::{hash_string, slice_memory_order_mut_or_fix, slice_memory_order_or_fix},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::{Ok, Result};
use ndarray::Ix4;
use onnx_extractor::{AttributeValue, OnnxOperation};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    #[default]
    Nearest,
    Linear,
    Cubic,
}

impl FromStr for Mode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "linear" => Self::Linear,
            "cubic" => Self::Cubic,
            _ => Self::Nearest,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateTransformationMode {
    HalfPixel,
    HalfPixelSymmetric,
    PytorchHalfPixel,
    AlignCorners,
    #[default]
    Asymmetric,
    TfCropAndResize,
}

impl FromStr for CoordinateTransformationMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "half_pixel" => Self::HalfPixel,
            "half_pixel_symmetric" => Self::HalfPixelSymmetric,
            "pytorch_half_pixel" => Self::PytorchHalfPixel,
            "align_corners" => Self::AlignCorners,
            "tf_crop_and_resize" => Self::TfCropAndResize,
            _ => Self::Asymmetric,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepAspectRatioPolicy {
    #[default]
    NotLarger,
    NotSmaller,
}

impl FromStr for KeepAspectRatioPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "not_smaller" => Ok(Self::NotSmaller),
            _ => Ok(Self::NotLarger),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NearestMode {
    #[default]
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

impl FromStr for NearestMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "round_prefer_ceil" => Ok(Self::RoundPreferCeil),
            "floor" => Ok(Self::Floor),
            "ceil" => Ok(Self::Ceil),
            _ => Ok(Self::RoundPreferFloor),
        }
    }
}

#[derive(Default)]
pub struct ResizeNode<T: Default> {
    x: u64,
    roi: Option<u64>,
    scales: Option<u64>,
    sizes: Option<u64>,

    o: u64,

    unique_id: UniqueId,

    antialias: i64,
    axes: Vec<usize>,
    mode: Mode,
    cubic_coeff_a: f32,
    exclude_outside: bool,
    extrapolation_value: f32,
    keep_aspect_ratio_policy: KeepAspectRatioPolicy,
    neares_mode: NearestMode,
    coordinate_transformation_mode: CoordinateTransformationMode,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for ResizeNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes();
        let inputs = &elem.inputs();
        let roi = inputs
            .get(1)
            .filter(|s| !s.is_empty())
            .map(|val| hash_string(val));
        let scales = inputs
            .get(2)
            .filter(|s| !s.is_empty())
            .map(|val| hash_string(val));
        let sizes = inputs
            .get(3)
            .filter(|s| !s.is_empty())
            .map(|val| hash_string(val));

        let mut resize = Self {
            x: u64::default(),
            roi: None,
            scales: None,
            sizes: None,

            o: u64::default(),
            unique_id: UniqueId::Resize,

            antialias: match attrs.get("antialias") {
                Some(av) => av
                    .as_int()
                    .ok_or_else(|| anyhow::anyhow!("Resize: antialias must be int"))?,
                None => 0,
            },
            axes: {
                match attrs.get("axes") {
                    Some(av) => av
                        .as_ints()
                        .ok_or_else(|| anyhow::anyhow!("Resize: axes must be ints"))?
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            mode: match attrs.get("mode") {
                Some(av) => {
                    let mode = av
                        .as_string()
                        .ok_or_else(|| anyhow::anyhow!("Resize: mode must be string"))?;
                    Mode::from_str(mode)
                        .map_err(|e| anyhow::anyhow!("Resize: invalid mode {mode:?}: {e}"))?
                }
                None => Mode::default(),
            },
            cubic_coeff_a: match attrs.get("cubic_coeff_a") {
                Some(av) => av
                    .as_float()
                    .ok_or_else(|| anyhow::anyhow!("Resize: cubic_coeff_a must be float"))?,
                None => 0.0f32,
            },
            exclude_outside: match attrs.get("exclude_outside") {
                Some(av) => av
                    .as_int()
                    .ok_or_else(|| anyhow::anyhow!("Resize: exclude_outside must be int"))?
                    != 0,
                None => false,
            },
            extrapolation_value: match attrs.get("extrapolation_value") {
                Some(av) => av
                    .as_float()
                    .ok_or_else(|| anyhow::anyhow!("Resize: extrapolation_value must be float"))?,
                None => 0.0f32,
            },
            keep_aspect_ratio_policy: match attrs.get("keep_aspect_ratio_policy") {
                Some(av) => {
                    let policy = av.as_string().ok_or_else(|| {
                        anyhow::anyhow!("Resize: keep_aspect_ratio_policy must be string")
                    })?;
                    KeepAspectRatioPolicy::from_str(policy).map_err(|e| {
                        anyhow::anyhow!("Resize: invalid keep_aspect_ratio_policy {policy:?}: {e}")
                    })?
                }
                None => KeepAspectRatioPolicy::default(),
            },
            neares_mode: match attrs.get("nearest_mode") {
                Some(av) => {
                    let mode = av
                        .as_string()
                        .ok_or_else(|| anyhow::anyhow!("Resize: nearest_mode must be string"))?;
                    NearestMode::from_str(mode)
                        .map_err(|e| anyhow::anyhow!("Resize: invalid nearest_mode {mode:?}: {e}"))?
                }
                None => NearestMode::default(),
            },
            coordinate_transformation_mode: match attrs.get("coordinate_transformation_mode") {
                Some(av) => {
                    let mode = av.as_string().ok_or_else(|| {
                        anyhow::anyhow!("Resize: coordinate_transformation_mode must be string")
                    })?;
                    CoordinateTransformationMode::from_str(mode).map_err(|e| {
                        anyhow::anyhow!(
                            "Resize: invalid coordinate_transformation_mode {mode:?}: {e}"
                        )
                    })?
                }
                None => CoordinateTransformationMode::default(),
            },
            next_node: None,
        };

        let x_id = hash_string(&elem.inputs()[0]);
        let o_id = hash_string(&elem.outputs()[0]);

        resize.add_inputs(x_id, roi, scales, sizes);
        resize.add_outputs(o_id);

        Ok(resize)
    }
}

impl<T: Default> ResizeNode<T> {
    pub fn new(
        antialias: i64,
        axes: Vec<usize>,
        mode: &str,
        cubic_coeff_a: f32,
        exclude_outside: bool,
        extrapolation_value: f32,
        keep_aspect_ratio_policy: &str,
        coordinate_transformation_mode: &str,
        neares_mode: &str,
    ) -> Self {
        Self {
            x: u64::default(),
            roi: None,
            scales: None,
            sizes: None,

            o: u64::default(),

            antialias,
            axes,
            mode: Mode::from_str(mode).unwrap_or_default(),
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy: KeepAspectRatioPolicy::from_str(keep_aspect_ratio_policy)
                .unwrap_or_default(),
            neares_mode: NearestMode::from_str(neares_mode).unwrap_or_default(),
            coordinate_transformation_mode: CoordinateTransformationMode::from_str(
                coordinate_transformation_mode,
            )
            .unwrap_or_default(),
            unique_id: UniqueId::Resize,
            next_node: None,
        }
    }

    pub fn add_inputs(
        &mut self,
        x: u64,
        roi: Option<u64>,
        scales: Option<u64>,
        sizes: Option<u64>,
    ) {
        self.x = x;
        self.roi = roi;
        self.scales = scales;
        self.sizes = sizes;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for ResizeNode<T> {
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
        let roi = self.roi.clone().unwrap_or_default();
        let scales = self.scales.clone().unwrap_or_default();
        let sizes = self.sizes.clone().unwrap_or_default();

        vec![self.x.clone(), roi, scales, sizes]
    }

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let empty = 0;
        let sizes_key = self.sizes.as_ref().unwrap_or(&empty);
        let scales_key = self.scales.as_ref().unwrap_or(&empty);

        let [x, sizes, scales, o] =
            omap.get_disjoint_mut([&self.x, sizes_key, scales_key, &self.o]);
        crate::debug_check_tensors!("ResizeNode", x => self.x, o => self.o);
        if self.sizes.is_some() {
            crate::debug_check_tensors!("ResizeNode", sizes => *sizes_key);
        }
        if self.scales.is_some() {
            crate::debug_check_tensors!("ResizeNode", scales => *scales_key);
        }
        let sizes = if self.sizes.is_some() {
            sizes.map(|val| &*val)
        } else {
            None
        };
        let scales = if self.scales.is_some() {
            scales.map(|val| &*val)
        } else {
            None
        };
        if let (Some(x), Some(out)) = (x, o) {
            x.resize(sizes, scales, &self.mode, out)?;
        }
        Ok(())
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "resize-{},{:?},{:?},{:?},{}",
            self.x, self.roi, self.scales, self.sizes, self.o
        );

        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap)?;
            }
        }
        Ok(())
    }
}

impl TypedArray {
    pub fn resize(
        &self,
        sizes: Option<&TypedArray>,
        scales: Option<&TypedArray>,
        mode: &Mode,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        match self {
            TypedArray::Float(x) => {
                let shape = x.shape();
                let hin = shape[shape.len() - 2];
                let win = shape[shape.len() - 1];
                let mut x_buf = x.clone();
                let in_sl = slice_memory_order_or_fix(&mut x_buf, "resize")?;

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
                let out_sl = slice_memory_order_mut_or_fix(out, "resize")?;

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
}
