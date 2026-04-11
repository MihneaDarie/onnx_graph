use std::any::Any;

use crate::{
    nodes::{node::Node, onnx_operation_trait::FromOnnxOperation, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;
use onnx_extractor::OnnxOperation;

#[derive(Default)]
pub struct SplitNode<T: Default> {
    input: String,
    split: String,

    o: Vec<String>,

    unique_id: UniqueId,

    axis: i64,
    num_outputs: i64,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> FromOnnxOperation for SplitNode<T> {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self> {
        let attrs = &elem.attributes;
        let mut split = Self {
            input: String::new(),
            split: String::new(),

            o: vec![],

            unique_id: UniqueId::Split,

            axis: match attrs.get("axis") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            num_outputs: match attrs.get("num_outputs") {
                Some(av) => av.as_int().unwrap(),
                None => 0,
            },
            next_node: None,
        };

        split.add_input_strings(elem.inputs[0].clone(), elem.inputs[1].clone());
        split.add_output_strings(elem.outputs.clone());

        Ok(split)
    }
}

impl<T: Default> SplitNode<T> {
    pub fn new(axis: i64, num_outputs: i64) -> Self {
        Self {
            input: String::new(),
            split: String::new(),

            o: vec![],
            axis,
            num_outputs,
            unique_id: UniqueId::Split,
            next_node: None,
        }
    }

    pub fn add_input_strings(&mut self, input: String, split: String) {
        self.input = input;
        self.split = split;
    }

    pub fn add_output_strings(&mut self, o: Vec<String>) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for SplitNode<T> {
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

    fn execute(&self, omap: &mut TensorMap) {
        let input = omap.get(&self.input);

        let split_sizes: Vec<i64> = if let Some(TypedArray::Int64(a)) = omap.get(&self.split) {
            a.iter().cloned().collect()
        } else if self.num_outputs > 0 {
            let input_ref = input.as_ref().unwrap();
            let axis = self.axis as usize;
            let dim = match input_ref {
                TypedArray::Float(a) => a.shape()[axis],
                _ => panic!("unsupported type"),
            };
            let chunk = dim / self.num_outputs as usize;
            vec![chunk as i64; self.num_outputs as usize]
        } else {
            panic!("SplitNode: no split tensor and no num_outputs");
        };

        match input {
            Some(input) => {
                let split_tensor = TypedArray::Int64(ndarray::Array1::from(split_sizes).into_dyn());
                let mut results = Vec::new();
                input.split(&split_tensor, self.axis, &mut results).unwrap();

                for (name, chunk) in self.o.iter().zip(results.into_iter()) {
                    omap.insert(name.clone(), chunk);
                }
            }
            None => panic!("SplitNode: missing input {}", self.input),
        }
    }

    fn output_names(&self) -> Vec<String> {
        self.o.clone()
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
        vec![self.input.clone(), self.split.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("split-{},{},{:?}", self.input, self.split, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) {
        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}

macro_rules! call_split_for_typed_array {
    ($self:expr, $axis:expr, $splits:expr, $outputs:expr, [$($variant:ident),+]) => {
        use ndarray::IxDyn;

        match $self {
            $(
                TypedArray::$variant(a) => split_variant!($variant, $axis, $splits, a, $outputs),
            )+
            _ => return Err(anyhow::anyhow!("unsupported type for split")),
        }
    };
}

macro_rules! split_variant {
    ($variant:ident, $axis:expr, $splits:expr, $a:expr, $outputs:expr) => {{
        let ndim = $a.ndim() as i64;
        let axis = if $axis < 0 {
            (ndim + $axis) as usize
        } else {
            $axis as usize
        };

        let mut offset = 0;
        for &size in $splits.iter() {
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

            $outputs.push(TypedArray::$variant(
                $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?)
                    .to_owned(),
            ));
            offset += size;
        }
    }};
}

impl TypedArray {
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
}
