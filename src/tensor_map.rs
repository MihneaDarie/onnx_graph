use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use crate::{nodes_utils::hash_string, typed_array::TypedArray};

#[derive(Default)]
pub struct TensorMap {
    inner: HashMap<u64, TypedArray>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn insert(&mut self, key: u64, val: TypedArray) {
        self.inner.insert(key, val.ensure_contiguous());
    }

    #[inline]
    pub fn get_disjoint_mut<const N: usize>(
        &mut self,
        key: [&u64; N],
    ) -> [Option<&mut TypedArray>; N] {
        self.inner.get_disjoint_mut(key)
    }

    #[inline]
    pub fn get(&self, key: &u64) -> Option<&TypedArray> {
        self.inner.get(key)
    }

    #[inline]
    pub fn get_from_str(&self, key: &str) -> Option<&TypedArray> {
        let id = hash_string(key);
        self.inner.get(&id)
    }

    #[inline]
    pub fn get_mut(&mut self, key: &u64) -> Option<&mut TypedArray> {
        self.inner.get_mut(key)
    }

    #[inline]
    pub fn remove(&mut self, key: &u64) -> Option<TypedArray> {
        self.inner.remove(key)
    }

    #[inline]
    pub fn contains_key(&self, key: &u64) -> bool {
        self.inner.contains_key(key)
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &TypedArray)> {
        self.inner.iter()
    }
}

pub struct UnsafeSendMut<T>(pub *mut T);
unsafe impl<T> Send for UnsafeSendMut<T> {}
unsafe impl<T> Sync for UnsafeSendMut<T> {}

impl<T> UnsafeSendMut<T> {
    pub unsafe fn as_mut(&self) -> &mut T {
        unsafe { &mut *self.0 }
    }
    pub unsafe fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}
