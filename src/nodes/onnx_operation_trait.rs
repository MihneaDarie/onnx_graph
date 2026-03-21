use anyhow::Result;

use onnx_extractor::OnnxOperation;

pub trait FromOnnxOperation: Sized {
    fn from_onnx_operation(elem: &OnnxOperation) -> Result<Self>;
}
