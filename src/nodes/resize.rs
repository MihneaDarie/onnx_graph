use std::{any::Any, collections::HashMap, str::FromStr};

use crate::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::{Ok, Result};
use onnx_extractor::AttributeValue;

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

impl<T: Default> FromHashMap for ResizeNode<T> {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self> {
        Ok(Self {
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
        })
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
