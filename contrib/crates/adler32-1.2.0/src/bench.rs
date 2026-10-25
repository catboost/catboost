use adler32::RollingAdler32;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use humansize::{file_size_opts, FileSize};
use rand::Rng;

fn bench_update_buffer(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("update_buffer");
    for &size in [512, 100 * 1024].iter() {
        let mut adler = RollingAdler32::new();
        let formatted_size = size.file_size(file_size_opts::BINARY).unwrap();
        let in_bytes = {
            let mut in_bytes = vec![0u8; size];
            rng.fill(&mut in_bytes[..]);
            in_bytes
        };

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(formatted_size),
            &in_bytes,
            |b, data| {
                b.iter(|| adler.update_buffer(data));
            },
        );
    }
}

criterion_group!(bench_default, bench_update_buffer);
criterion_main!(bench_default);
