use std::{any::Any, collections::HashMap, str::FromStr};

use crate::{
    nodes::{hash_trait::FromHashMap, node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Clone, Copy)]
pub struct Conv2D {
    pub pad: usize,
    pub stride: usize,
}

use anyhow::{Ok, Result};
use onnx_extractor::AttributeValue;
use saker_rs::activations::Activation;

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
pub struct ConvNode<T: Default> {
    x: String,
    w: String,
    b: Option<String>,

    o: String,

    activation: Activation,

    unique_id: UniqueId,

    auto_pad: AutoPad,
    kernel_shape: Vec<usize>,
    group: i64,
    pads: Vec<usize>,
    strides: Vec<usize>,
    dilations: Vec<usize>,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromHashMap for ConvNode<T> {
    fn from_hashmap(
        attrs: &std::collections::HashMap<String, AttributeValue>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            x: String::new(),
            w: String::new(),
            b: None,
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
                        .to_vec()
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
                        .to_vec()
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
                        .to_vec()
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
                        .to_vec()
                        .iter()
                        .map(|&val| val as usize)
                        .collect(),
                    None => vec![],
                }
            },
            group: {
                match attrs.get("groups") {
                    Some(av) => av.as_int().unwrap(),
                    None => 0,
                }
            },
            unique_id: UniqueId::Conv,
            activation: Activation::None,
            next_node: None,
        })
    }
}

impl<T: Default> ConvNode<T> {
    pub fn new(
        auto_pad: &str,
        kernel_shape: Vec<usize>,
        group: i64,
        pads: Vec<usize>,
        strides: Vec<usize>,
        dilations: Vec<usize>,
        activation: Activation,
    ) -> Self {
        Self {
            x: String::new(),
            w: String::new(),
            b: None,
            o: String::new(),
            auto_pad: AutoPad::from_str(auto_pad).unwrap(),
            kernel_shape,
            group,
            pads,
            strides,
            dilations,
            unique_id: UniqueId::Conv,
            next_node: None,
            activation: activation,
        }
    }

    pub fn add_input_strings(&mut self, x: String, w: String, b: Option<String>) {
        self.x = x;
        self.w = w;
        self.b = b;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }

    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }
}

impl<T: Default + 'static> Node<T> for ConvNode<T> {
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
        let b = self.b.clone().unwrap_or(String::from(""));
        vec![self.x.clone(), self.w.clone(), b]
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn execute(&self, omap: &mut TensorMap) {
        let def = &String::from("");
        let b = self.b.as_ref().unwrap_or(def);

        let [x, w, b, o] = omap.get_disjoint_mut([&self.x, &self.w, b, &self.o]);
        let x = &*x.unwrap();
        let w = &*w.unwrap();
        let b = b.map(|b| &*b);
        match o {
            Some(result) => {
                let cfg = Conv2D {
                    pad: self.pads.first().copied().unwrap_or(0),
                    stride: self.strides.first().copied().unwrap_or(1),
                };
                x.conv(w, b, &cfg, result, self.activation).unwrap();
            }
            _ => panic!("ConvNode: missing input(s) - x={} w={}", self.x, self.w),
        }
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("conv-{},{},{:?},{}", self.x, self.w, self.b, self.o);

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
        let [x, w, o] = omap.get_disjoint_mut([&self.x, &self.w, &self.o]);
        let x = x.map(|arr| &*arr);
        let w = w.map(|arr| &*arr);

        if let (Some(x), Some(w), Some(o)) = (x, w, o) {
            if let (Some(x_shape), Some(w_shape)) = (x.shape(), w.shape()) {
                let batch = x_shape[0];
                let cout = w_shape[0];
                let kh = w_shape[2];
                let kw = w_shape[3];
                let hin = x_shape[2];
                let win = x_shape[3];

                let ph = self.pads.first().copied().unwrap_or(0);
                let pw = self.pads.get(1).copied().unwrap_or(ph);
                let sh = self.strides.first().copied().unwrap_or(1);
                let sw = self.strides.get(1).copied().unwrap_or(sh);
                let dh = self.dilations.first().copied().unwrap_or(1);
                let dw = self.dilations.get(1).copied().unwrap_or(dh);

                let hout = (hin + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
                let wout = (win + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

                let out_shape = &[batch, cout, hout, wout];
                *o = TypedArray::empty_with_others_type(x, out_shape);
            }
        }
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
