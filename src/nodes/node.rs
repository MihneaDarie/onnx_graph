use std::{any::Any, collections::HashMap};

use anyhow::{Ok, Result};

use crate::{nodes::unique_ids::UniqueId, tensor_map::TensorMap, typed_array::TypedArray};

pub trait Node<T: Default + 'static>: Send + Sync {
    fn execute(&self, omap: &mut TensorMap) -> anyhow::Result<()>;

    fn determine_output_shape(&mut self, omap: &mut TensorMap) -> anyhow::Result<()>;

    fn print(&self);

    fn self_count(&self) -> usize {
        let mut count = 1;
        if let Some(children) = &self.get_next() {
            for child in children.iter() {
                count += child.self_count();
            }
        }
        count
    }

    fn get_next(&self) -> Option<&Vec<Box<dyn Node<T>>>>;
    fn get_next_mut(&mut self) -> Option<&mut Vec<Box<dyn Node<T>>>>;
    fn set_next(&mut self, next: Option<Vec<Box<dyn Node<T>>>>);
    fn take_next(&mut self) -> Option<Vec<Box<dyn Node<T>>>>;

    fn input_hashes(&self) -> Vec<u64>;
    fn output_hashes(&self) -> Vec<u64>;
    fn get_unique_id(&self) -> UniqueId;
    fn get_unique_id_mut(&mut self) -> UniqueId;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn optimize_further(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub fn pass_node<T: Default + 'static>(
    node: &dyn Node<T>,
    omap: &mut TensorMap,
) -> anyhow::Result<()> {
    let mut current: &dyn Node<T> = node;
    loop {
        current.execute(omap)?;
        match current.get_next() {
            Some(children) if children.len() == 1 => {
                current = children[0].as_ref();
            }
            Some(children) => {
                for child in children {
                    pass_node(child.as_ref(), omap)?;
                }
                return Ok(());
            }
            None => return Ok(()),
        }
    }
}

pub fn insert_node<T: Default + 'static>(
    node: &mut dyn Node<T>,
    next: Box<dyn Node<T>>,
) -> Result<()> {
    let mut current_ptr: *mut dyn Node<T> = node;
    loop {
        unsafe {
            let current = &mut *current_ptr;
            if let Some(children) = current.get_next_mut() {
                current_ptr = children[0].as_mut() as *mut dyn Node<T>;
            } else {
                current.set_next(Some(vec![next]));
                return Ok(());
            }
        }
    }
}

pub fn print_node<T: Default + 'static>(node: &dyn Node<T>) {
    let mut current: &dyn Node<T> = node;
    loop {
        current.print();
        match current.get_next() {
            Some(children) if children.len() == 1 => {
                current = children[0].as_ref();
            }
            Some(children) => {
                for child in children {
                    print_node(child.as_ref());
                }
                return;
            }
            None => return,
        }
    }
}
