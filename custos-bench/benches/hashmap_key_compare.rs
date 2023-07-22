use std::{
    alloc::Layout,
    collections::{hash_map::RandomState, HashMap, HashSet},
    hash::{BuildHasher, BuildHasherDefault, Hash},
    hint::black_box,
};

use criterion::{criterion_group, criterion_main, Criterion};
use custos::{Ident, IdentHasher};

#[derive(Default)]
pub struct NoHasher {
    hash: u64,
}

impl std::hash::Hasher for NoHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("NoHasher only hashes usize.")
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.hash = i as u64;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        // println!("hash: {i}");
        self.hash = i;
    }
}

fn build_ident_hm<S: BuildHasher + Default>() -> (HashMap<Ident, Vec<f64>, S>, Vec<Ident>) {
    let mut hm = HashMap::<Ident, Vec<f64>, S>::default();
    let mut idents = Vec::new();
    for idx in 0..1000 {
        let len = fastrand::usize(1..10) * 1000;
        let ident = Ident { idx, len };
        idents.push(ident);
        hm.insert(ident, vec![10.0; len]);
    }
    (hm, idents)
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct Location<'a> {
    file: &'a str,
    line: u32,
    col: u32,
}

impl PartialEq for Location<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // self.file == other.file &&
        self.line == other.line && self.col == other.col
    }
}

impl<'a> std::hash::Hash for Location<'a> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // self.file.hash(state);
        let bytes = self.file.as_bytes();
        // let res: u64 = ((self.line as u64) << 40) | ((self.col as u64) << 32);
        let res = ((self.line as u64) << 8 | self.col as u64) << 32;
        let file = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let res = res | file as u64;
        res.hash(state);
    }
}

fn build_location_hm<'a, S: BuildHasher + Default>(
) -> (HashMap<Location<'a>, Vec<f64>, S>, Vec<Location<'a>>) {
    let mut hm = HashMap::<Location<'a>, Vec<f64>, S>::default();
    let mut locations = Vec::new();
    let mut hashes = HashSet::new();

    let filenames = [
        "main.rs",
        "cache.rs",
        "module_comb.rs",
        "hash.rs",
        "hashmap_key_compare.rs",
        "fasthash.rs",
        "custos.rs",
        "ident.rs",
        "caller_cache.rs",
        "stack_array.rs",
        "graph.rs",
        "mod.rs",
        "ptr_conv.rs",
    ];

    for idx in 0..1000 {
        let len = fastrand::usize(1..10) * 1000;
        let filename_idx = fastrand::usize(0..filenames.len());

        let line = fastrand::u32(70..3000);
        let col = fastrand::u32(4..255);

        let location = Location {
            file: filenames[filename_idx],
            line,
            col,
        };
        let mut hasher = custos::module_comb::LocationHasher::default();
        location.hash(&mut hasher);

        use std::hash::Hasher;

        let hash = hasher.finish();
        // println!("hash: {hash}");
        if hashes.contains(&hash) {
            panic!("hash collision");
        }
        hashes.insert(hash);

        locations.push(location);
        hm.insert(location, vec![10.0; len]);
    }

    (hm, locations)
}

fn build_idx_hm<S: BuildHasher + Default>() -> (HashMap<usize, Vec<f64>, S>, Vec<usize>) {
    let mut hm = HashMap::<usize, Vec<f64>, S>::default();
    let mut idents = Vec::new();
    for idx in 0..1000 {
        let len = fastrand::usize(1..10) * 1000;
        idents.push(idx);
        hm.insert(idx, vec![10.0; len]);
    }
    (hm, idents)
}

fn bench_location_key_hash(c: &mut Criterion) {
    let (hm, locations) = build_location_hm::<BuildHasherDefault<NoHasher>>();

    c.bench_function("bench_location_nohasher", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..locations.len());
            let location = locations[idx];
            let _ = black_box(hm.get(&location));
        })
    });

    let (hm, locations) = build_location_hm::<RandomState>();

    c.bench_function("bench_location_key_hash", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..locations.len());
            let location = &locations[idx];
            let _ = black_box(hm.get(location));
        })
    });

    let (hm, locations) = build_location_hm::<BuildHasherDefault<LocationHasher>>();

    c.bench_function("bench_location_key_hash_location_hasher", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..locations.len());
            let location = &locations[idx];
            let _ = black_box(hm.get(location));
        })
    });
}

fn bench_ident_key_hash(c: &mut Criterion) {
    let (hm, idents) = build_ident_hm::<BuildHasherDefault<IdentHasher>>();

    c.bench_function("bench_ident_key_hash", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..idents.len());
            let ident = idents[idx];
            let _ = black_box(hm.get(&ident));
        })
    });

    let (hm, idents) = build_ident_hm::<RandomState>();

    c.bench_function("bench_ident_default", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..idents.len());
            let ident = idents[idx];
            let _ = black_box(hm.get(&ident));
        })
    });

    let (hm, idxs) = build_idx_hm::<BuildHasherDefault<NoHasher>>();

    c.bench_function("bench_nohasher", |bench| {
        bench.iter(|| {
            let idx = fastrand::usize(0..idxs.len());
            let idx = idxs[idx];
            let _ = black_box(hm.get(&idx));
        })
    });
}

fn bench_alloc_speed(c: &mut Criterion) {
    c.bench_function("bench_alloc_speed", |bench| {
        bench.iter(|| {
            let ptr = black_box(unsafe {
                std::alloc::alloc(Layout::from_size_align(1000000, 4).unwrap())
            });
            unsafe { std::alloc::dealloc(ptr, Layout::from_size_align(1000000, 4).unwrap()) };
        })
    });

    c.bench_function("bench_alloc_speed_multiple", |bench| {
        bench.iter(|| {
            for _ in 0..1000000 / 10000 {
                let ptr = black_box(unsafe {
                    std::alloc::alloc(Layout::from_size_align(10000, 4).unwrap())
                });
                unsafe { std::alloc::dealloc(ptr, Layout::from_size_align(10000, 4).unwrap()) };
            }
        })
    });
}

criterion_group!(benches, bench_location_key_hash, bench_ident_key_hash);
criterion_main!(benches);
