use std::{any::Any, collections::HashMap};

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use anyhow::Result;

#[derive(Default)]
pub struct SliceNode<T: Default> {
    data: String,
    o: String,

    starts: String,
    ends: String,
    axes: String,

    unique_id: UniqueId,

    next_node: Option<Vec<Box<dyn Node<T>>>>,
}

impl<T: Default> SliceNode<T> {
    pub fn new() -> Self {
        Self {
            data: String::new(),
            starts: String::new(),
            ends: String::new(),
            axes: String::new(),
            o: String::new(),
            unique_id: UniqueId::Slice,
            next_node: None,
        }
    }
    pub fn add_input_strings(&mut self, data: String, starts: String, ends: String, axes: String) {
        self.data = data;
        self.starts = starts;
        self.ends = ends;
        self.axes = axes;
    }

    pub fn add_output_strings(&mut self, o: String) {
        self.o = o;
    }
}

impl<T: Default + 'static> Node<T> for SliceNode<T> {
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
        let [data, starts, ends, axes, o] =
            omap.get_disjoint_mut([&self.data, &self.starts, &self.ends, &self.axes, &self.o]);
        let data = &*data.unwrap();
        let starts = &*starts.unwrap();
        let ends = &*ends.unwrap();
        let axes = &*axes.unwrap();

        match o {
            Some(result) => {
                data.slice(starts, ends, axes, result).unwrap();
            }
            _ => panic!(
                "SliceNode: missing input(s) - data={} starts={} ends={} axes={}",
                self.data, self.starts, self.ends, self.axes
            ),
        }
    }

    fn output_names(&self) -> Vec<String> {
        vec![self.o.clone()]
    }

    fn input_names(&self) -> Vec<String> {
        vec![
            self.data.clone(),
            self.starts.clone(),
            self.ends.clone(),
            self.axes.clone(),
        ]
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

    fn print(&self) {
        if let Some(list) = &self.next_node {
            print!("{}-", list.len());
        }
        println!(
            "slice-{},{},{},{},{}",
            self.data, self.starts, self.ends, self.axes, self.o
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
        let [data, starts, ends, axes, o] =
            omap.get_disjoint_mut([&self.data, &self.starts, &self.ends, &self.axes, &self.o]);
        let data = data.map(|arr| &*arr);
        let starts = starts.map(|arr| &*arr);
        let ends = ends.map(|arr| &*arr);
        let axes = axes.map(|arr| &*arr);

        if let (Some(data), Some(o)) = (data, o)
            && let Some(in_shape) = data.shape()
            && let (
                Some(TypedArray::I64(starts)),
                Some(TypedArray::I64(ends)),
                Some(TypedArray::I64(axes)),
            ) = (starts, ends, axes)
        {
            let mut out_shape = in_shape.to_vec();

            for i in 0..axes.len() {
                let axis = axes[i] as usize;
                let dim_size = in_shape[axis] as i64;

                let start = {
                    let s = starts[i];
                    if s < 0 {
                        (dim_size + s).max(0)
                    } else {
                        s.min(dim_size)
                    }
                } as usize;

                let end = {
                    let e = ends[i];
                    if e < 0 {
                        (dim_size + e).max(0)
                    } else {
                        e.min(dim_size)
                    }
                } as usize;

                out_shape[axis] = end - start;
            }

            *o = TypedArray::empty_with_others_type(data, &out_shape);
        }

        if let Some(list) = &mut self.next_node {
            for next in list {
                next.determine_output_shape(omap);
            }
        }
    }
}
