use std::mem;

use ndarray::ArrayD;

use crate::typed_array::TypedArray;

pub fn hash_string(str: &str) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    str.hash(&mut hasher);
    hasher.finish()
}

#[macro_export]
macro_rules! debug_check_tensors {
    ($node:expr, $( $field:ident => $id:expr ),+ $(,)?) => {
        if cfg!(debug_assertions) {
            $(
                if $field.is_none() {
                    anyhow::bail!(concat!($node, ": missing ", stringify!($field), " (id={})"), $id);
                }
            )+
        }
    };
}

#[inline(always)]
pub fn ensure_contiguous_in_place(arr: &mut TypedArray) -> anyhow::Result<()> {
    if matches!(arr, TypedArray::Undefined) {
        anyhow::bail!("cannot make Undefined array contiguous");
    }
    let taken = mem::replace(arr, TypedArray::Undefined);
    *arr = taken.ensure_contiguous();
    Ok(())
}

#[inline(always)]
pub fn slice_memory_order_view<'a, T>(
    arr: &'a ArrayD<T>,
    ctx: &str,
) -> anyhow::Result<&'a [T]> {
    arr.as_slice_memory_order()
        .ok_or_else(|| anyhow::anyhow!("{ctx}: array not contiguous"))
}

#[inline(always)]
pub fn slice_memory_order_or_fix<'a, T: Clone>(
    arr: &'a mut ArrayD<T>,
    ctx: &str,
) -> anyhow::Result<&'a [T]> {
    if !arr.is_standard_layout() {
        *arr = arr.as_standard_layout().into_owned();
    }
    arr.as_slice_memory_order()
        .ok_or_else(|| anyhow::anyhow!("{ctx}: array not contiguous after fix"))
}

#[inline(always)]
pub fn slice_memory_order_mut_or_fix<'a, T: Clone>(
    arr: &'a mut ArrayD<T>,
    ctx: &str,
) -> anyhow::Result<&'a mut [T]> {
    if !arr.is_standard_layout() {
        *arr = arr.as_standard_layout().into_owned();
    }
    arr.as_slice_memory_order_mut()
        .ok_or_else(|| anyhow::anyhow!("{ctx}: array not contiguous after fix"))
}
