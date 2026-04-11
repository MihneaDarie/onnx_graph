use std::{any::Any, collections::HashMap};

use anyhow::{Ok, Result};

use crate::{nodes::unique_ids::UniqueId, tensor_map::TensorMap, typed_array::TypedArray};

pub trait Node<T: Default + 'static>: Send + Sync {
    fn execute(&self, omap: &mut TensorMap);

    fn determine_output_shape(&mut self, omap: &mut TensorMap);

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

    fn input_names(&self) -> Vec<String>;
    fn output_names(&self) -> Vec<String>;
    fn get_unique_id(&self) -> UniqueId;
    fn get_unique_id_mut(&mut self) -> UniqueId;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn optimize_further(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

pub fn pass_node<T: Default + 'static>(node: &dyn Node<T>, omap: &mut TensorMap) {
    let mut current: &dyn Node<T> = node;
    loop {
        current.execute(omap);
        match current.get_next() {
            Some(children) if children.len() == 1 => {
                current = children[0].as_ref();
            }
            Some(children) => {
                for child in children {
                    pass_node(child.as_ref(), omap);
                }
                return;
            }
            None => return,
        }
    }
}

pub fn insert_node<T: Default + 'static>(
    node: &mut dyn Node<T>,
    next: Box<dyn Node<T>>,
) -> Result<()> {
    let mut current: &mut dyn Node<T> = node;
    loop {
        if current.get_next_mut().is_some() {
            let children = current.get_next_mut().unwrap();
            current = children[0].as_mut();
        } else {
            current.set_next(Some(vec![next]));
            return Ok(());
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
