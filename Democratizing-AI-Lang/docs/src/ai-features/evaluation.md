# Model Evaluation

Democratising provides comprehensive model evaluation capabilities. This guide explains how to effectively evaluate and analyze model performance using various metrics and techniques.

## Basic Evaluation

### Model Assessment

```rust
use democratising::prelude::*;

fn evaluate_model(model: &Model, test_loader: &DataLoader) -> Result<()> {
    // Set model to evaluation mode
    model.eval();

    // Disable gradient computation
    no_grad(|| {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for (x, y) in test_loader {
            // Forward pass
            let predictions = model.forward(&x)?;
            let loss = cross_entropy_loss(&predictions, &y)?;

            // Compute accuracy
            let pred_classes = predictions.argmax(-1)?;
            correct += (&pred_classes == &y).sum()?.item();
            total += y.size(0);

            total_loss += loss.item();
        }

        // Print metrics
        println!(
            "Test Loss: {:.4}, Accuracy: {:.2}%",
            total_loss / total as f32,
            100.0 * correct as f32 / total as f32
        );

        Ok(())
    })
}
```

### Multiple Metrics

```rust
fn compute_metrics(
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<HashMap<String, f32>> {
    let mut metrics = HashMap::new();

    // Accuracy
    let accuracy = compute_accuracy(predictions, targets)?;
    metrics.insert("accuracy".into(), accuracy);

    // Precision, Recall, F1 (multi-class)
    let (precision, recall, f1) = compute_precision_recall_f1(predictions, targets)?;
    metrics.insert("precision".into(), precision);
    metrics.insert("recall".into(), recall);
    metrics.insert("f1".into(), f1);

    // ROC AUC (for binary classification)
    if predictions.size(-1) == 2 {
        let auc = compute_roc_auc(predictions, targets)?;
        metrics.insert("roc_auc".into(), auc);
    }

    Ok(metrics)
}
```

## Advanced Evaluation

### Cross Validation

```rust
fn cross_validate(
    model_fn: impl Fn() -> Result<Model>,
    dataset: &Dataset,
    k_folds: usize,
) -> Result<Vec<f32>> {
    let mut scores = Vec::new();

    // Create k folds
    let folds = dataset.k_fold_split(k_folds)?;

    for fold_idx in 0..k_folds {
        // Create new model instance
        let mut model = model_fn()?;

        // Get train/val data for this fold
        let (train_data, val_data) = folds.get_fold(fold_idx)?;

        // Train model
        train_model(&mut model, &train_data)?;

        // Evaluate on validation fold
        let score = evaluate_model(&model, &val_data)?;
        scores.push(score);
    }

    Ok(scores)
}
```

### Confusion Matrix

```rust
struct ConfusionMatrix {
    matrix: Tensor,
    labels: Vec<String>,
}

impl ConfusionMatrix {
    fn new(num_classes: usize) -> Result<Self> {
        Ok(ConfusionMatrix {
            matrix: Tensor::zeros(&[num_classes, num_classes])?,
            labels: (0..num_classes).map(|i| i.to_string()).collect(),
        })
    }

    fn update(&mut self, predictions: &Tensor, targets: &Tensor) -> Result<()> {
        for (pred, target) in predictions.iter()?.zip(targets.iter()?) {
            self.matrix[[target as usize, pred as usize]] += 1;
        }
        Ok(())
    }

    fn normalize(&self) -> Result<Tensor> {
        let sums = self.matrix.sum(1)?.unsqueeze(1)?;
        &self.matrix / &sums
    }

    fn plot(&self) -> Result<()> {
        // Plot confusion matrix using plotters
        plot_confusion_matrix(&self.matrix, &self.labels)
    }
}
```

### Error Analysis

```rust
fn analyze_errors(
    model: &Model,
    data: &DataLoader,
    num_examples: usize,
) -> Result<()> {
    let mut errors = Vec::new();

    model.eval();
    no_grad(|| {
        for (x, y) in data {
            let predictions = model.forward(&x)?;
            let pred_classes = predictions.argmax(-1)?;

            // Find misclassified examples
            let incorrect = (&pred_classes != &y).nonzero()?;
            for idx in incorrect.iter()? {
                errors.push(ErrorExample {
                    input: x[idx].clone(),
                    prediction: pred_classes[idx].item(),
                    target: y[idx].item(),
                    confidence: predictions[idx].softmax(-1)?,
                });

                if errors.len() >= num_examples {
                    return Ok(());
                }
            }
        }
        Ok(())
    })?;

    // Analyze and visualize errors
    analyze_error_patterns(&errors)
}
```

## Performance Analysis

### Model Profiling

```rust
fn profile_model(model: &Model, input_shape: &[usize]) -> Result<()> {
    // Create profiler
    let mut profiler = ModelProfiler::new()
        .with_memory_tracking(true)
        .with_cuda_events(true);

    // Profile forward pass
    profiler.start()?;
    let x = Tensor::randn(input_shape)?;
    let _ = model.forward(&x)?;
    let stats = profiler.stop()?;

    // Print statistics
    println!("Layer-wise statistics:");
    for (name, layer_stats) in stats.layer_stats {
        println!(
            "{}: compute={:.2}ms, memory={:.2}MB",
            name,
            layer_stats.compute_ms,
            layer_stats.memory_mb
        );
    }

    Ok(())
}
```

### Throughput Measurement

```rust
fn measure_throughput(
    model: &Model,
    batch_size: usize,
    num_iterations: usize,
) -> Result<f32> {
    model.eval();
    cuda::synchronize()?;

    let start = std::time::Instant::now();
    let x = Tensor::randn(&[batch_size, 3, 224, 224])?;

    no_grad(|| {
        for _ in 0..num_iterations {
            let _ = model.forward(&x)?;
        }
        cuda::synchronize()?;
        Ok(())
    })?;

    let elapsed = start.elapsed();
    let throughput = (batch_size * num_iterations) as f32 / elapsed.as_secs_f32();

    Ok(throughput)
}
```

## Visualization

### Training Curves

```rust
fn plot_training_history(history: &TrainingHistory) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new("training_history.png", (800, 600))?;
    let mut chart = ChartBuilder::on(&root)
        .set_title("Training History")
        .build_cartesian_2d(
            0..history.epochs(),
            0f32..history.max_value()?,
        )?;

    // Plot training loss
    chart.draw_series(LineSeries::new(
        history.loss_history().iter().enumerate(),
        &RED,
    ))?;

    // Plot validation loss
    chart.draw_series(LineSeries::new(
        history.val_loss_history().iter().enumerate(),
        &BLUE,
    ))?;

    Ok(())
}
```

### Feature Visualization

```rust
fn visualize_features(
    model: &Model,
    layer_name: &str,
    input: &Tensor,
) -> Result<()> {
    // Get intermediate activations
    let features = model.get_intermediate_features(layer_name, input)?;

    // Reduce dimensionality for visualization
    let reduced = dimensionality_reduction(&features, 2)?;

    // Plot features
    plot_scatter(
        &reduced,
        "Feature Space Visualization",
        "Dimension 1",
        "Dimension 2",
    )
}
```

## Best Practices

### Evaluation Pipeline

1. Create evaluation workflow:
```rust
fn evaluation_pipeline() -> Result<()> {
    // Load test data
    let test_loader = DataLoader::new(test_dataset)?
        .batch_size(32)
        .num_workers(4)
        .build()?;

    // Create metrics
    let mut metrics = MetricCollection::new()
        .add(Accuracy::new())
        .add(Precision::new())
        .add(Recall::new())
        .add(F1Score::new());

    // Evaluate
    model.eval();
    no_grad(|| {
        for batch in test_loader {
            let output = model.forward(&batch.x)?;
            metrics.update(&output, &batch.y)?;
        }
    })?;

    // Log results
    println!("Results: {}", metrics);

    Ok(())
}
```

2. Robust evaluation:
```rust
fn robust_evaluation() -> Result<()> {
    // Multiple random seeds
    let mut scores = Vec::new();
    for seed in seeds {
        set_seed(seed);
        let score = evaluate_model(&model, &test_data)?;
        scores.push(score);
    }

    // Compute statistics
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let std = compute_std(&scores);
    println!("Score: {:.4} ± {:.4}", mean, std);

    Ok(())
}
```

### Performance Considerations

1. Efficient evaluation:
```rust
fn efficient_eval() -> Result<()> {
    // Use appropriate batch size
    let batch_size = determine_optimal_batch_size()?;

    // Enable CUDA graphs for repeated inference
    let graph = CudaGraph::new()?;
    graph.capture(|| {
        model.forward(&static_input)
    })?;

    // Evaluate with graph
    for batch in test_loader {
        graph.replay()?;
    }

    Ok(())
}
```

2. Memory management:
```rust
fn manage_eval_memory() -> Result<()> {
    // Clear cache before evaluation
    cuda::empty_cache()?;

    // Use streaming evaluation for large datasets
    let loader = StreamingDataLoader::new(test_dataset)?
        .batch_size(32)
        .prefetch(2)
        .build()?;

    for batch in loader {
        // Process batch and free memory
        process_batch(&batch)?;
        cuda::empty_cache()?;
    }

    Ok(())
}
```

## Next Steps

- Learn about [Hyperparameter Tuning](hyperparameter-tuning.md)
- Explore [Model Deployment](deployment.md)
- Study [Performance Profiling](profiling.md)
