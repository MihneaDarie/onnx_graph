# onnx_graph

A minimal ONNX graph parser and inference engine written in Rust. Takes an `.onnx` file, builds an internal linked list of nodes, and runs forward inference by executing each node in order and writing results into a contiguous tensor map.

> Currently supports the subset of ONNX operators required for YOLOv8 inference.

## How it works

1. Parses the `.onnx` file and builds a linked list where each node corresponds to an ONNX operator
2. Initializes a `TensorMap` — a `HashMap<String, TypedArray>` wrapper that ensures all arrays are contiguous in memory
3. Runs inference by walking the linked list, each node reads its inputs from the `TensorMap` and writes its output back into it

## Usage

```toml
[dependencies]
onnx_graph = "0.1.0"
```

```rust
use onnx_graph::{graph::GraphForm, typed_array::TypedArray};

// Load model and initialize tensor map
let (mut graph, mut omap) =
    GraphForm::<f32>::from_onnx_file("models/yolov8n.onnx")?;

// Run inference
graph.pass(&mut omap, &input.into_dyn());
```

The `TensorMap` automatically ensures contiguous memory layout on every insert:

```rust
pub struct TensorMap {
    inner: HashMap<String, TypedArray>,
}
```

Tensors are stored as `TypedArray`, a typed enum over ndarray's `ArrayD`:

```rust
pub enum TypedArray {
    Undefined,
    F32(ArrayD<f32>),
    U8(ArrayD<u8>),
    // other pytorch data types
}
```

## License

MIT