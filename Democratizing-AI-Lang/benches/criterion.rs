use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use democratising::{
    nn::{
        activation::{ReLU, Sigmoid},
        layer::{Conv2D, Dense, Dropout, MaxPool2D},
        loss::CrossEntropyLoss,
        optimizer::Adam,
        Model,
    },
    tensor::Tensor,
    Device,
};

fn tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test different matrix sizes
    for size in [128, 256, 512, 1024, 2048].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create input tensors
        let a = Tensor::randn(&[n, n]).unwrap();
        let b = Tensor::randn(&[n, n]).unwrap();

        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", n), &n, |b, _| {
            b.iter(|| black_box(&a).matmul(black_box(&b)).unwrap())
        });

        // Element-wise operations
        group.bench_with_input(BenchmarkId::new("add", n), &n, |b, _| {
            b.iter(|| black_box(&a + &b).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("multiply", n), &n, |b, _| {
            b.iter(|| black_box(&a * &b).unwrap())
        });

        // Reductions
        group.bench_with_input(BenchmarkId::new("sum", n), &n, |b, _| {
            b.iter(|| black_box(&a).sum(&[0, 1]).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("mean", n), &n, |b, _| {
            b.iter(|| black_box(&a).mean(&[0, 1]).unwrap())
        });
    }

    group.finish();
}

fn neural_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_network");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test different batch sizes
    for batch_size in [16, 32, 64, 128].iter() {
        let n = *batch_size;
        group.throughput(Throughput::Elements(n as u64));

        // Create model
        let mut model = Model::new();
        model.add_layer(Conv2D::new(1, 32, 3, ReLU::default()));
        model.add_layer(MaxPool2D::new(2));
        model.add_layer(Conv2D::new(32, 64, 3, ReLU::default()));
        model.add_layer(MaxPool2D::new(2));
        model.add_layer(Dense::new(64 * 5 * 5, 512, ReLU::default()));
        model.add_layer(Dropout::new(0.5));
        model.add_layer(Dense::new(512, 10, Sigmoid::default()));
        model.set_optimizer(Adam::new(0.001));
        model.set_loss(CrossEntropyLoss::default());

        // Create input data
        let input = Tensor::randn(&[n, 1, 28, 28]).unwrap();
        let target = Tensor::randn(&[n, 10]).unwrap();

        // Forward pass
        group.bench_with_input(BenchmarkId::new("forward", n), &n, |b, _| {
            b.iter(|| black_box(&model).forward(black_box(&input)).unwrap())
        });

        // Backward pass
        group.bench_with_input(BenchmarkId::new("backward", n), &n, |b, _| {
            b.iter(|| {
                black_box(&mut model)
                    .train_step(black_box(&input), black_box(&target))
                    .unwrap()
            })
        });
    }

    group.finish();
}

fn gpu_operations(c: &mut Criterion) {
    // Only run GPU benchmarks if CUDA is available
    if !cfg!(feature = "gpu") {
        return;
    }

    let mut group = c.benchmark_group("gpu_operations");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test different matrix sizes
    for size in [1024, 2048, 4096].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create CPU tensors
        let a = Tensor::randn(&[n, n]).unwrap();
        let b = Tensor::randn(&[n, n]).unwrap();

        // Create GPU tensors
        let a_gpu = a.to_device(Device::GPU(0)).unwrap();
        let b_gpu = b.to_device(Device::GPU(0)).unwrap();

        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul_cpu", n), &n, |b, _| {
            b.iter(|| black_box(&a).matmul(black_box(&b)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("matmul_gpu", n), &n, |b, _| {
            b.iter(|| black_box(&a_gpu).matmul(black_box(&b_gpu)).unwrap())
        });

        // Element-wise operations
        group.bench_with_input(BenchmarkId::new("add_cpu", n), &n, |b, _| {
            b.iter(|| black_box(&a + &b).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("add_gpu", n), &n, |b, _| {
            b.iter(|| black_box(&a_gpu + &b_gpu).unwrap())
        });

        // Memory transfers
        group.bench_with_input(BenchmarkId::new("to_gpu", n), &n, |b, _| {
            b.iter(|| black_box(&a).to_device(Device::GPU(0)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("to_cpu", n), &n, |b, _| {
            b.iter(|| black_box(&a_gpu).to_device(Device::CPU).unwrap())
        });
    }

    group.finish();
}

fn autodiff(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test different sizes
    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create tensors
        let x = Tensor::randn(&[n, n]).unwrap().requires_grad(true);
        let y = Tensor::randn(&[n, n]).unwrap().requires_grad(true);

        // Forward operations
        let z = (&x * &y).unwrap() + &x.sum(&[1]).unwrap().reshape(&[-1, 1]).unwrap();
        let loss = z.mean(&[0, 1]).unwrap();

        // Backward pass
        group.bench_with_input(BenchmarkId::new("backward", n), &n, |b, _| {
            b.iter(|| {
                black_box(&loss).backward().unwrap();
                black_box(&x).zero_grad();
                black_box(&y).zero_grad();
            })
        });
    }

    group.finish();
}

fn distributed_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_training");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));

    // Test different number of workers
    for num_workers in [2, 4, 8].iter() {
        let n = *num_workers;
        group.throughput(Throughput::Elements(n as u64));

        // Create shared model
        let model = Model::new();
        let model = std::sync::Arc::new(std::sync::Mutex::new(model));

        // Create worker threads
        let handles: Vec<_> = (0..n)
            .map(|_| {
                let model = std::sync::Arc::clone(&model);
                std::thread::spawn(move || {
                    // Simulate worker computation
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let _guard = model.lock().unwrap();
                })
            })
            .collect();

        // Benchmark synchronization
        group.bench_with_input(BenchmarkId::new("sync", n), &n, |b, _| {
            b.iter(|| {
                for handle in handles.iter() {
                    black_box(handle.thread().id());
                }
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = tensor_operations, neural_network, gpu_operations, autodiff, distributed_training
}
criterion_main!(benches);
