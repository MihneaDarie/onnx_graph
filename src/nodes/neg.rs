use std::any::Any;

use onnx_extractor::OnnxOperation;

use crate::{
    impl_typed_singleopfunction_with_the_same_output_type_as_the_output,
    nodes::{node::Node, unique_ids::UniqueId},
    nodes_utils::hash_string,
    tensor_map::TensorMap,
    typed_array::TypedArray,
};

#[derive(Default)]
pub struct NegNode<T: Default> {
    pub x: u64,

    pub o: u64,

    unique_id: UniqueId,

    pub next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> NegNode<T> {
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut neg = Self {
            x: u64::default(),
            o: u64::default(),
            unique_id: UniqueId::Neg,
            next_node: None,
        };
        let x_id = hash_string(&elem.inputs()[0]);
        let o_id = hash_string(&elem.outputs()[0]);
        neg.add_inputs(x_id);
        neg.add_outputs(o_id);
        neg
    }

    pub fn add_inputs(&mut self, x: u64) {
        self.x = x;
    }

    pub fn add_outputs(&mut self, o: u64) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for NegNode<T> {
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

    fn input_hashes(&self) -> Vec<u64> {
        vec![self.x.clone()]
    }

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!("neg-{},{}", self.x, self.o);
        if let Some(next) = &self.next_node {
            next.iter().for_each(|v| v.print());
        }
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>> {
        self.next_node.as_ref()
    }

    fn output_hashes(&self) -> Vec<u64> {
        vec![self.o.clone()]
    }

    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|val| &*val);
        crate::debug_check_tensors!("NegNode", x => self.x, o => self.o);
        if let (Some(x), Some(out)) = (x, o) {
            x.neg(out)?;
        }
        Ok(())
    }

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()> {
        let [x, o] = omap.get_disjoint_mut([&self.x, &self.o]);
        let x = x.map(|arr| &*arr);

        if let (Some(x), Some(o)) = (x, o)
            && let Some(in_shape) = x.shape()
        {
            *o = TypedArray::empty_with_others_type(x, in_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap)?;
            }
        }
        Ok(())
    }
}

use std::ops::Neg;

impl TypedArray {
    impl_typed_singleopfunction_with_the_same_output_type_as_the_output!(
        neg,
        neg,
        [Float, Double, Int32, Int64, Int8, Int16],
        [Uint8, Uint16, Uint32, Uint64]
    );
}
