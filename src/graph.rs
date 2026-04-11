use std::collections::{HashMap, HashSet};

use crate::{
    nodes::{
        add::AddNode,
        and::AndNode,
        argmax::ArgMaxNode,
        cast::CastNode,
        concat::ConcatNode,
        constant_of_shape::ConstantOfShapeNode,
        conv::ConvNode,
        cos::CosNode,
        div::DivNode,
        equal::EqualNode,
        expand::ExpandNode,
        flatten::FlattenNode,
        gather::GatherNode,
        gemm::GemmNode,
        greater::GreaterNode,
        greater_or_equal::GreaterOrEqualNode,
        is_nan::IsNanNode,
        less::LessNode,
        less_or_equal::LessOrEqualNode,
        mat_mul::MatMulNode,
        max_pool::MaxPoolNode,
        mul::MulNode,
        neg::NegNode,
        node::{Node, insert_node, pass_node},
        onnx_operation_trait::FromOnnxOperation,
        pow::PowNode,
        range::RangeNode,
        reduce_mean::ReduceMeanNode,
        relu::ReluNode,
        reshape::ReshapeNode,
        resize::ResizeNode,
        shape::ShapeNode,
        sigmoid::SigmoidNode,
        sin::SinNode,
        slice::SliceNode,
        soft_max::SoftMaxNode,
        split::SplitNode,
        sqrt::SqrtNode,
        sub::SubNode,
        transpose::TransposeNode,
        unique_ids::UniqueId,
        unsqueeze::UnsquezeeNode,
        where_op::WhereNode,
    },
    tensor_map::TensorMap,
    typed_array::{TypedArray, TypedArrayDiscriminants},
};
use anyhow::Ok;
use ndarray::{ArrayD, IxDyn};
use onnx_extractor::{OnnxModel, OnnxOperation};
use saker_rs::activations::Activation;

#[derive(Default)]
pub struct GraphForm<T: Default> {
    inputs: Vec<String>,
    // nodes: Vec<Box<dyn Node<T>>>,
    pub nodes: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default + 'static> GraphForm<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, node: Box<dyn Node<T>>) {
        if let Some(next) = &mut self.nodes {
            insert_node(next[0].as_mut(), node).unwrap();
        } else {
            self.nodes = Some(vec![node]);
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

    pub fn self_count(&self) -> usize {
        if let Some(nodes) = &self.nodes {
            nodes.iter().map(|n| n.self_count()).sum()
        } else {
            0
        }
    }

    pub fn load_data_arrays(onnx: &OnnxModel) -> TensorMap {
        let mut map = TensorMap::new();

        onnx.get_output_tensors().iter().for_each(|tensor| {
            let shape = tensor.shape();
            if tensor.data().is_ok() {
                map.insert(tensor.name().to_string(), TypedArray::from_tensor(tensor));
            } else if shape.iter().any(|&d| d < 0) {
                map.insert(tensor.name().to_string(), TypedArray::Undefined);
            } else if !shape.is_empty() {
                map.insert(
                    tensor.name().to_string(),
                    TypedArray::from_tensor_empty(tensor, shape),
                );
            } else {
                map.insert(tensor.name().to_string(), TypedArray::Undefined);
            }
        });

        onnx.operations.iter().for_each(|s| {
            s.outputs.iter().for_each(|out| {
                map.insert(out.to_string(), TypedArray::Undefined);
            });
        });

        onnx.tensor_names().iter().for_each(|t| {
            if let Some(tensor) = onnx.get_tensor(t) {
                let shape = tensor.shape();

                if tensor.data().is_ok() {
                    map.insert(tensor.name().to_string(), TypedArray::from_tensor(&tensor));
                } else if shape.iter().any(|&d| d < 0) {
                    map.insert(tensor.name().to_string(), TypedArray::Undefined);
                } else if !shape.is_empty() {
                    map.insert(
                        tensor.name().to_string(),
                        TypedArray::from_tensor_empty(tensor, shape),
                    );
                } else {
                    map.insert(tensor.name().to_string(), TypedArray::Undefined);
                }
            }
        });

        onnx.get_input_tensors().iter().for_each(|tensor| {
            map.insert(tensor.name().to_string(), TypedArray::Undefined);
        });

        map
    }

    fn get_specifc_node_from_operation(
        elem: &OnnxOperation,
        omap: &mut TensorMap,
    ) -> anyhow::Result<Box<dyn Node<T>>> {
        let res: Box<dyn Node<T>> = match elem.op_type.as_str() {
            "Concat" => Box::new(ConcatNode::from_onnx_operation(elem)?),
            "Gather" => Box::new(GatherNode::from_onnx_operation(elem)?),
            "Conv" => Box::new(ConvNode::from_onnx_operation(elem)?),
            "Gemm" => Box::new(GemmNode::from_onnx_operation(elem)?),
            "Resize" => Box::new(ResizeNode::from_onnx_operation(elem)?),
            "Transpose" => Box::new(TransposeNode::from_onnx_operation(elem)?),
            "MaxPool" => Box::new(MaxPoolNode::from_onnx_operation(elem)?),

            "Flatten" => Box::new(FlattenNode::from_onnx_operation(elem)?),
            "Where" => Box::new(WhereNode::new(elem)),
            "Unsqueeze" => Box::new(UnsquezeeNode::new(elem)),

            "ConstantOfShape" | "Constant" => {
                Box::new(ConstantOfShapeNode::from_onnx_operation(elem)?)
            }

            "ReduceMean" => Box::new(ReduceMeanNode::from_onnx_operation(elem)?),

            "Cast" => Box::new(CastNode::from_onnx_operation(elem)?),
            "MatMul" => Box::new(MatMulNode::from_onnx_operation(elem)?),

            "Expand" => Box::new(ExpandNode::new(elem)),

            "Range" => Box::new(RangeNode::new(elem)),

            "Sigmoid" => Box::new(SigmoidNode::new(elem)),
            "Relu" => Box::new(ReluNode::new(elem)),

            "Sub" => Box::new(SubNode::new(elem)),
            "Add" => Box::new(AddNode::new(elem)),
            "Mul" => Box::new(MulNode::new(elem)),
            "Div" => Box::new(DivNode::new(elem)),

            "And" => Box::new(AndNode::new(elem)),

            "Sin" => Box::new(SinNode::new(elem)),
            "Cos" => Box::new(CosNode::new(elem)),

            "LessOrEqual" => Box::new(LessOrEqualNode::new(elem)),
            "Less" => Box::new(LessNode::new(elem)),
            "GreaterOrEqual" => Box::new(GreaterOrEqualNode::new(elem)),
            "Greater" => Box::new(GreaterNode::new(elem)),
            "Equal" => Box::new(EqualNode::new(elem)),

            "Sqrt" => Box::new(SqrtNode::new(elem)),

            "IsNaN" => Box::new(IsNanNode::new(elem)),

            "Pow" => Box::new(PowNode::new(elem)),

            "Neg" => Box::new(NegNode::new(elem)),

            "ArgMax" => Box::new(ArgMaxNode::from_onnx_operation(elem)?),
            "Softmax" => Box::new(SoftMaxNode::from_onnx_operation(&elem)?),
            "Split" => Box::new(SplitNode::from_onnx_operation(elem)?),
            "Reshape" => Box::new(ReshapeNode::from_onnx_operation(elem)?),
            "Shape" => Box::new(ShapeNode::from_onnx_operation(elem)?),
            "Slice" => Box::new(SliceNode::new(elem)),
            _ => {
                panic!("Unsupported node with name {}", elem.name)
            }
        };
        Ok(res)
    }

    pub fn from_onnx_file(onnx_file_path: &str) -> anyhow::Result<(Self, TensorMap)> {
        let onnx = OnnxModel::load_from_file(onnx_file_path)?;

        let mut ret = Self::new();
        ret.inputs = onnx.inputs.clone();

        let mut map = Self::load_data_arrays(&onnx);

        // onnx.execution_order()?.into_iter().for_each(|elem| {
        for elem in onnx.execution_order()? {
            ret.insert(Self::get_specifc_node_from_operation(elem, &mut map)?);
        }

        Ok((ret, map))
    }

    pub fn set_input(&self, omap: &mut TensorMap, input_name: &str, data: TypedArray) {
        if !self.inputs.contains(&String::from(input_name)) {
            println!("No such input called {}", input_name);
        }
        match omap.get_mut(input_name) {
            Some(inner) => {
                *inner = data;
            }
            None => {
                println!("No such input called {}", input_name);
            }
        }
    }

    pub fn determine_output_shape<const N: usize>(
        &mut self,
        omap: &mut TensorMap,
        inputs_info: [(&str, TypedArrayDiscriminants, &[usize]); N],
    ) {
        for (name, discriminant, shape) in inputs_info {
            if !self.inputs.contains(&String::from(name)) {
                println!("!!! No such input called {name} !!!");
            }
            omap.insert(
                name.to_string(),
                TypedArray::empty_from_discriminant(discriminant, shape).ensure_contiguous(),
            );
        }

        if let Some(start) = &mut self.nodes {
            for next in start {
                next.determine_output_shape(omap);
            }
        }
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

    pub fn pass(&self, omap: &mut TensorMap) {
        self.inputs.iter().for_each(|input| {
            let input_array = omap.get(input);
            match input_array {
                Some(array) => match array {
                    TypedArray::Undefined => {
                        println!("Undefined input tensor: <{input}>, you must call GraphForm::set_input function for this specific input");
                    }
                    _ => {}
                },
                None => println!("Couldn't find input tensor with name {}", input),
            }
        });
        if let Some(nodes) = &self.nodes {
            for node in nodes {
                pass_node(node.as_ref(), omap);
            }
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
