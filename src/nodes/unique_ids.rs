#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UniqueId {
    Add,
    Concat,
    Conv,
    Gemm,
    Div,
    MaxPool,
    Mul,
    Reshape,
    Resize,
    Slice,
    Softmax,
    Split,
    Sub,
    Transpose,
    
    ArgMax,
    Shape,
    Gather,

    //Activation
    Sigmoid,
    Silu,
    Relu,

    #[default]
    Undefined,
}
