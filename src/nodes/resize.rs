use std::{any::Any, collections::HashMap, str::FromStr};

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::{Ok, Result};
use ndarray::Ix4;
use onnx_extractor::{AttributeValue, OnnxOperation};
use rayon::{iter::{IndexedParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

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
    x: String,
    roi: Option<String>,
    scales: Option<String>,
    sizes: Option<String>,

    o: String,

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
        let attrs = &elem.attributes;
        let inputs = &elem.inputs;
        let roi = inputs.get(1).filter(|s| !s.is_empty()).cloned();
        let scales = inputs.get(2).filter(|s| !s.is_empty()).cloned();
        let sizes = inputs.get(3).filter(|s| !s.is_empty()).cloned();

        let mut resize = Self {
            x: String::new(),
            roi: None,
            scales: None,
            sizes: None,

            o: String::new(),
            unique_id: UniqueId::Resize,

            antialias: match attrs.get("antialias") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            axes: {
                match attrs.get("axes") {
                    Some(av) => av
                        .as_ints()
                        .unwrap()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            mode: match attrs.get("mode") {
                Some(av) => Mode::from_str(av.as_string().unwrap()).unwrap(),
                None => Mode::default(),
            },
            cubic_coeff_a: match attrs.get("cubic_coeff_a") {
                Some(av) => av.as_float().unwrap(),
                None => 0.0f32,
            },
            exclude_outside: match attrs.get("exclude_outside") {
                Some(av) => av.as_int().unwrap() != 0,
                None => false,
            },
            extrapolation_value: match attrs.get("extrapolation_value") {
                Some(av) => av.as_float().unwrap(),
                None => 0.0f32,
            },
            keep_aspect_ratio_policy: match attrs.get("keep_aspect_ratio_policy") {
                Some(av) => KeepAspectRatioPolicy::from_str(av.as_string().unwrap()).unwrap(),
                None => KeepAspectRatioPolicy::default(),
            },
            neares_mode: match attrs.get("nearest_mode") {
                Some(av) => NearestMode::from_str(av.as_string().unwrap()).unwrap(),
                None => NearestMode::default(),
            },
            coordinate_transformation_mode: match attrs.get("coordinate_transformation_mode") {
                Some(av) => {
                    CoordinateTransformationMode::from_str(av.as_string().unwrap()).unwrap()
                }
                None => CoordinateTransformationMode::default(),
            },
            next_node: None,
        };

        resize.add_input_strings(inputs[0].clone(), roi, scales, sizes);
        resize.add_output_strings(elem.outputs[0].clone());

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
            x: String::new(),
            roi: None,
            scales: None,
            sizes: None,

            o: String::new(),

            antialias,
            axes,
            mode: Mode::from_str(mode).unwrap(),
            cubic_coeff_a,
            exclude_outside,
            extrapolation_value,
            keep_aspect_ratio_policy: KeepAspectRatioPolicy::from_str(keep_aspect_ratio_policy)
                .unwrap(),
            neares_mode: NearestMode::from_str(neares_mode).unwrap(),
            coordinate_transformation_mode: CoordinateTransformationMode::from_str(
                coordinate_transformation_mode,
            )
            .unwrap(),
            unique_id: UniqueId::Resize,
            next_node: None,
        }
    }

    pub fn add_input_strings(
        &mut self,
        x: String,
        roi: Option<String>,
        scales: Option<String>,
        sizes: Option<String>,
    ) {
        self.x = x;
        self.roi = roi;
        self.scales = scales;
        self.sizes = sizes;
    }

    pub fn add_output_strings(&mut self, o: String) {
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

    fn input_names(&self) -> Vec<String> {
        let roi = self.roi.clone().unwrap_or(String::from(""));
        let scales = self.scales.clone().unwrap_or(String::from(""));
        let sizes = self.sizes.clone().unwrap_or(String::from(""));

        vec![self.x.clone(), roi, scales, sizes]
    }

    fn execute(&self, omap: &mut TensorMap) {
        let empty = String::from("");
        let sizes = self.sizes.as_ref().unwrap_or(&empty);
        let scales = self.scales.as_ref().unwrap_or(&empty);

        let [x, sizes, scales, o] = omap.get_disjoint_mut([&self.x, sizes, scales, &self.o]);
        let x = &*x.unwrap();
        let sizes = sizes.as_deref();
        let scales = scales.as_deref();

        match o {
            Some(result) => {
                x.resize(sizes, scales, &self.mode, result).unwrap();
            }
            None => panic!("ResizeNode: missing input x={}", self.x),
        }
    }

    fn output_names(&self) -> Vec<String> {
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
}
