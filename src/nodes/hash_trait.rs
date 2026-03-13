use anyhow::Result;
use std::collections::HashMap;

use onnx_extractor::AttributeValue;

pub trait FromHashMap: Sized {
    fn from_hashmap(attrs: &HashMap<String, AttributeValue>) -> Result<Self>;
}
