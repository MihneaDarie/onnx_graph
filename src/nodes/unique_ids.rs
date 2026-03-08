#[repr(u8)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UniqueId {
    Add,
    Concat,
    Conv,
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
    
    //Activation
    Sigmoid,
    Silu,

    #[default]
    Undefined,
}
