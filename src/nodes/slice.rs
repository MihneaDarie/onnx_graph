use std::any::Any;

use crate::{
    nodes::{node::Node, unique_ids::UniqueId},
    tensor_map::TensorMap,
    typed_array::TypedArray,
};
use onnx_extractor::OnnxOperation;

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
    pub fn new(elem: &OnnxOperation) -> Self {
        let mut slice = Self {
            data: String::new(),
            starts: String::new(),
            ends: String::new(),
            axes: String::new(),
            o: String::new(),
            unique_id: UniqueId::Slice,
            next_node: None,
        };
        let input = &elem.inputs;
        slice.add_input_strings(
            input[0].clone(),
            input[1].clone(),
            input[2].clone(),
            input[3].clone(),
        );
        slice.add_output_strings(elem.outputs[0].clone());
        slice
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
                Some(TypedArray::Int64(starts)),
                Some(TypedArray::Int64(ends)),
                Some(TypedArray::Int64(axes)),
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

macro_rules! call_slice_for_typed_array {
    ($self:expr, $axes:expr, $starts:expr, $ends:expr, $o:expr, [$($variant:ident),+]) => {
        use ndarray::IxDyn;

        match $self {
            $(
                TypedArray::$variant(a) => slice_variant!($variant, $axes, $starts, $ends, a, $o),
            )+
            TypedArray::Bool(a) => {
                use ndarray::ArrayD;
                let ndim = a.ndim();
                let mut slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
                    .map(|_| ndarray::SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    })
                    .collect();

                let mut out_shape = a.shape().to_vec();

                for i in 0..$axes.len() {
                    let axis = $axes[i] as usize;
                    let dim_size = a.shape()[axis] as i64;

                    let start = {
                        let s = $starts[i];
                        if s < 0 {
                            (dim_size + s).max(0)
                        } else {
                            s.min(dim_size)
                        }
                    } as usize;

                    let end = {
                        let e = $ends[i];
                        if e < 0 {
                            (dim_size + e).max(0)
                        } else {
                            e.min(dim_size)
                        }
                    } as usize;

                    out_shape[axis] = end - start;

                    slice_info[axis] = ndarray::SliceInfoElem::Slice {
                        start: start as isize,
                        end: Some(end as isize),
                        step: 1,
                    };
                }

                let needs_alloc = match &*$o {
                    TypedArray::Bool(out) => out.shape() != out_shape.as_slice(),
                    _ => true,
                };
                if needs_alloc {
                    *$o = TypedArray::Bool(ArrayD::from_elem(IxDyn(&out_shape), false));
                }

                let view = a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?);

                if let TypedArray::Bool(out) = $o {
                    let dst = out.as_slice_memory_order_mut().unwrap();
                    for (d, s) in dst.iter_mut().zip(view.iter()) {
                        *d = *s;
                    }
                }
            }
            _ => return Err(anyhow::anyhow!("unsupported type for slice")),
        }
    };
}

macro_rules! slice_variant {
    ($variant:ident, $axes:expr, $starts:expr, $ends:expr, $a:expr, $o:expr) => {{
        use ndarray::ArrayD;
        let ndim = $a.ndim();
        let mut slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
            .map(|_| ndarray::SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();

        let mut out_shape = $a.shape().to_vec();

        for i in 0..$axes.len() {
            let axis = $axes[i] as usize;
            let dim_size = $a.shape()[axis] as i64;

            let start = {
                let s = $starts[i];
                if s < 0 {
                    (dim_size + s).max(0)
                } else {
                    s.min(dim_size)
                }
            } as usize;

            let end = {
                let e = $ends[i];
                if e < 0 {
                    (dim_size + e).max(0)
                } else {
                    e.min(dim_size)
                }
            } as usize;

            out_shape[axis] = end - start;

            slice_info[axis] = ndarray::SliceInfoElem::Slice {
                start: start as isize,
                end: Some(end as isize),
                step: 1,
            };
        }

        let needs_alloc = match &*$o {
            TypedArray::$variant(out) => out.shape() != out_shape.as_slice(),
            _ => true,
        };
        if needs_alloc {
            *$o = TypedArray::$variant(ArrayD::zeros(IxDyn(&out_shape)));
        }

        let view = $a.slice(ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)?);

        if let TypedArray::$variant(out) = $o {
            let dst = out.as_slice_memory_order_mut().unwrap();
            for (d, s) in dst.iter_mut().zip(view.iter()) {
                *d = *s;
            }
        }
    }};
}

impl TypedArray {
    pub fn slice(
        &self,
        starts: &TypedArray,
        ends: &TypedArray,
        axes: &TypedArray,
        o: &mut TypedArray,
    ) -> anyhow::Result<()> {
        let starts = match starts {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("starts must be I64")),
        };
        let ends = match ends {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("ends must be I64")),
        };
        let axes = match axes {
            TypedArray::Int64(s) => s,
            _ => return Err(anyhow::anyhow!("axes must be I64")),
        };

        call_slice_for_typed_array!(
            self,
            axes,
            starts,
            ends,
            o,
            [Float, Double, Int32, Int64, Uint8, Uint16, Uint32, Uint64]
        );

        Ok(())
    }
}
