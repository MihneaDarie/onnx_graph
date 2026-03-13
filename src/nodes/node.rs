use std::{any::Any, collections::HashMap};

use anyhow::{Ok, Result};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    nodes::unique_ids::UniqueId,
    tensor_map::{TensorMap, UnsafeSendMut},
};

pub fn fuse_silu() {}

pub trait Node<T: Default + 'static>: Send + Sync {
    fn pass(&self, omap: &mut TensorMap) {
        self.execute(omap);

        if let Some(children) = self.get_next() {
            if children.len() == 1 {
                children[0].pass(omap);
            } else {
                let ptr = UnsafeSendMut(omap as *mut TensorMap);

                children.par_iter().for_each(|branch: &Box<dyn Node<T>>| {
                    let map = unsafe { ptr.as_mut() };
                    branch.pass(map);
                });
            }
        }
    }
    fn execute(&self, omap: &mut TensorMap);

    fn determine_output_shape(&mut self, omap: &mut TensorMap);

    fn print(&self);
    fn self_count(&self, count: usize) -> usize;
    fn insert(&mut self, next: Box<dyn Node<T>>) -> Result<()>;

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>>;
    fn get_next_mut(&mut self) -> Option<&mut Vec<Box<dyn Node<T>>>>;
    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>);
    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>>;

    fn input_names(&self) -> Vec<String>;
    fn output_names(&self) -> Vec<String>;
    fn get_unique_id(&self) -> UniqueId;
    fn get_unique_id_mut(&mut self) -> UniqueId;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn optimize_further(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}
