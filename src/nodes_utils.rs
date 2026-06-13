use std::hash::{DefaultHasher, Hash, Hasher};

pub fn hash_string(str: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    str.hash(&mut hasher);
    let id = hasher.finish();

    id
}
