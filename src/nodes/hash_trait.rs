use anyhow::Result;
use std::collections::HashMap;

use onnx_extractor::{AttributeValue, OnnxOperation};

pub trait FromHashMap: Sized {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>, elem: &OnnxOperation) -> Result<Self>;
}
