use std::collections::{HashMap, HashSet};

use crate::{
    nodes::{
        add::AddNode, concat::ConcatNode, conv::ConvNode, div::DivNode, hash_trait::FromHashMap,
        max_pool::MaxPoolNode, mul::MulNode, node::Node, reshape::ReshapeNode, resize::ResizeNode,
        sigmoid::SigmoidNode, slice::SliceNode, soft_max::SoftMaxNode, split::SplitNode,
        sub::SubNode, transpose::TransposeNode, unique_ids::UniqueId,
    },
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Ok;
use ndarray::ArrayD;
use onnx_extractor::OnnxModel;
use saker_rs::activations::Activation;

pub struct GraphForm<T: Default> {
    // nodes: Vec<Box<dyn Node<T>>>,
    nodes: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default + 'static> GraphForm<T> {
    pub fn new() -> Self {
        Self { nodes: None }
    }

    pub fn insert(&mut self, node: Box<dyn Node<T>>) {
        if let Some(next) = &mut self.nodes {
            next[0].insert(node).unwrap();
        } else {
            self.nodes = Some(vec![node])
        }
    }

    pub fn print(&self) {
        if let Some(list) = &self.nodes {
            println!("{}", list.len());
        }
        if let Some(next) = &self.nodes {
            next.iter().for_each(|v| v.print());
        }
    }

    fn self_count(&self, count: usize) -> usize {
        if let Some(next) = &self.nodes {
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

    pub fn load_data_arrays(onnx: &OnnxModel) -> TensorMap {
        let mut map = TensorMap::new();

        onnx.operations.iter().for_each(|s| {
            s.outputs.iter().for_each(|out| {
                map.insert(out.to_string(), TypedArray::Undefined);
            });
        });

        onnx.tensor_names().iter().for_each(|t| {
            if let Some(tensor) = onnx.get_tensor(t) {
                let typed = if tensor.data().is_ok() {
                    TypedArray::from_tensor(&tensor)
                } else {
                    TypedArray::from_tensor_empty(tensor)
                };
                map.insert(tensor.name().to_string(), typed);
            }
        });

        map
    }

    pub fn from_onnx_file(onnx_file_path: &str) -> anyhow::Result<(Self, TensorMap)> {
        let onnx = OnnxModel::load_from_file(onnx_file_path)?;
        let mut ret = Self::new();
        let map = Self::load_data_arrays(&onnx);

        onnx.execution_order()?
            .into_iter()
            .for_each(|elem| match elem.op_type.as_str() {
                "Concat" => {
                    let mut concat = ConcatNode::from_hashmap(&elem.attributes).unwrap();
                    concat.add_input_strings(elem.inputs.clone());
                    concat.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(concat));
                }
                "Sigmoid" => {
                    let mut sigmoid = SigmoidNode::new();
                    sigmoid.add_input_strings(elem.inputs[0].clone());
                    sigmoid.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sigmoid));
                }
                "Conv" => {
                    let mut conv = ConvNode::from_hashmap(&elem.attributes).unwrap();
                    let inputs = &elem.inputs;
                    let b = inputs.get(2).cloned();
                    conv.add_input_strings(inputs[0].clone(), inputs[1].clone(), b);
                    conv.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(conv));
                }
                "Resize" => {
                    println!("{:?}", elem.attributes);
                    let inputs = &elem.inputs;
                    let roi = inputs.get(1).filter(|s| !s.is_empty()).cloned();
                    let scales = inputs.get(2).filter(|s| !s.is_empty()).cloned();
                    let sizes = inputs.get(3).filter(|s| !s.is_empty()).cloned();

                    let mut resize = ResizeNode::from_hashmap(&elem.attributes).unwrap();
                    resize.add_input_strings(inputs[0].clone(), roi, scales, sizes);
                    resize.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(resize));
                }
                "Transpose" => {
                    let mut trans = TransposeNode::from_hashmap(&elem.attributes).unwrap();
                    trans.add_input_strings(elem.inputs[0].clone());
                    trans.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(trans));
                }
                "Sub" => {
                    let mut sub = SubNode::new();
                    sub.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    sub.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(sub));
                }
                "MaxPool" => {
                    let mut max_pool = MaxPoolNode::from_hashmap(&elem.attributes).unwrap();
                    max_pool.add_input_strings(elem.inputs[0].clone());
                    max_pool.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(max_pool));
                }
                "Div" => {
                    let mut div = DivNode::new();
                    div.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    div.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(div));
                }
                "Softmax" => {
                    let mut soft_max = SoftMaxNode::from_hashmap(&elem.attributes).unwrap();
                    soft_max.add_input_strings(elem.inputs[0].clone());
                    soft_max.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(soft_max));
                }
                "Split" => {
                    let mut split = SplitNode::from_hashmap(&elem.attributes).unwrap();
                    split.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    split.add_output_strings(elem.outputs.clone());
                    ret.insert(Box::new(split));
                }
                "Add" => {
                    let mut add = AddNode::new();
                    add.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    add.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(add));
                }
                "Mul" => {
                    let mut mul = MulNode::new();
                    mul.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    mul.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(mul));
                }
                "Reshape" => {
                    let mut reshape = ReshapeNode::from_hashmap(&elem.attributes).unwrap();
                    reshape.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
                    reshape.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(reshape));
                }
                "Slice" => {
                    let mut slice = SliceNode::new();
                    let input = &elem.inputs;
                    slice.add_input_strings(
                        input[0].clone(),
                        input[1].clone(),
                        input[2].clone(),
                        input[3].clone(),
                    );
                    slice.add_output_strings(elem.outputs[0].clone());
                    ret.insert(Box::new(slice));
                }
                _ => {}
            });

        Ok((ret, map))
    }

    pub fn rearange_for_parallel_branches(&mut self) {}

    pub fn optimize(&mut self) {
        let mut convs: HashSet<String> = HashSet::new();
        let mut sigmoids: HashMap<String, String> = HashMap::new();
        let mut fuse_targets: HashMap<String, String> = HashMap::new();
        let mut mul_to_remove: HashSet<String> = HashSet::new();

        let mut conv_to_mul: HashMap<String, String> = HashMap::new();

        if let Some(nodes) = &self.nodes {
            collect(
                nodes,
                &mut convs,
                &mut sigmoids,
                &mut fuse_targets,
                &mut mul_to_remove,
            );
        }

        for (conv_out, sigmoid_out) in &sigmoids {
            if let Some(mul_out) = fuse_targets.get(sigmoid_out) {
                conv_to_mul.insert(conv_out.clone(), mul_out.clone());
            }
        }

        if !fuse_targets.is_empty()
            && let Some(nodes) = &mut self.nodes
        {
            apply(nodes, &conv_to_mul, &fuse_targets, &mul_to_remove);
        }
    }

    pub fn pass(&self, omap: &mut TensorMap, input: &ArrayD<f32>) {
        omap.insert("images".to_string(), TypedArray::F32(input.clone()));

        if let Some(nodes) = &self.nodes {
            nodes.iter().for_each(|val| val.pass(omap));
        }
    }
}

fn collect<T: Default + 'static>(
    nodes: &[Box<dyn Node<T>>],
    convs: &mut HashSet<String>,
    sigmoids: &mut HashMap<String, String>,
    fuse_targets: &mut HashMap<String, String>,
    mul_to_remove: &mut HashSet<String>,
) {
    for node in nodes {
        match node.get_unique_id() {
            UniqueId::Conv => {
                convs.insert(node.output_names()[0].clone());
            }
            UniqueId::Sigmoid => {
                let inp = node.input_names()[0].clone();
                if convs.contains(&inp) {
                    sigmoids.insert(inp, node.output_names()[0].clone());
                }
            }
            UniqueId::Mul => {
                let inputs = node.input_names();
                for (conv_out, sigmoid_out) in sigmoids.iter() {
                    if inputs.contains(conv_out) && inputs.contains(sigmoid_out) {
                        let mul_out = node.output_names()[0].clone();
                        fuse_targets.insert(sigmoid_out.clone(), mul_out.clone());
                        mul_to_remove.insert(mul_out);
                    }
                }
            }
            _ => {}
        }

        if let Some(children) = node.get_next() {
            collect(children, convs, sigmoids, fuse_targets, mul_to_remove);
        }
    }
}

fn apply<T: Default + 'static>(
    nodes: &mut [Box<dyn Node<T>>],
    conv_to_mul: &HashMap<String, String>,
    fuse_targets: &HashMap<String, String>,
    mul_to_remove: &HashSet<String>,
) {
    for node in nodes.iter_mut() {
        if node.get_unique_id() == UniqueId::Conv {
            let conv_out = node.output_names()[0].clone();
            if let Some(mul_out) = conv_to_mul.get(&conv_out)
                && let Some(conv) = node.as_any_mut().downcast_mut::<ConvNode<T>>()
            {
                conv.set_activation(Activation::Silu);
                conv.add_output_strings(mul_out.clone());
            }
        }

        let mut children = match node.take_next() {
            Some(c) => c,
            None => continue,
        };

        let mut i = 0;
        while i < children.len() {
            let uid = children[i].get_unique_id();
            let out = children[i].output_names()[0].clone();

            let should_remove = (uid == UniqueId::Sigmoid && fuse_targets.contains_key(&out))
                || (uid == UniqueId::Mul && mul_to_remove.contains(&out));

            if should_remove {
                let mut removed = children.remove(i);
                if let Some(grandchildren) = removed.take_next() {
                    for (j, gc) in grandchildren.into_iter().enumerate() {
                        children.insert(i + j, gc);
                    }
                }
            } else {
                i += 1;
            }
        }

        apply(&mut children, conv_to_mul, fuse_targets, mul_to_remove);
        node.set_next(Some(children));
    }
}
