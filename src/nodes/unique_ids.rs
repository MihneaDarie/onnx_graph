#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UniqueId {
    Concat,
    Conv,
    Gemm,
    MaxPool,
    Reshape,
    Resize,
    Slice,
    Softmax,
    Split,
    Transpose,

    Sub,
    Mul,
    Div,
    Add,
    And,
    Pow,
    Sin,

    ArgMax,
    Shape,
    Gather,

    Neg,

    //Activation
    Sigmoid,
    Silu,
    Relu,

    #[default]
    Undefined,
}
