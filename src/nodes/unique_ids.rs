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

    Unsqueeze,
    Squeeze,
    Cast,
    Flatten,
    Where,
    ConstantOfShape,
    Range,

    LessOrEqual,
    Less,
    GreaterOrEqual,
    Greater,
    Equal,

    Sub,
    Mul,
    Div,
    Add,
    And,
    Pow,
    Sin,
    Cos,

    Sqrt,
    Expand,

    IsNan,

    ArgMax,
    Shape,
    Gather,

    Neg,
    Abs,
    Not,

    //Activation
    Sigmoid,
    Erf,
    Silu,
    Relu,
    LeakyRelu,

    #[default]
    Undefined,
}
