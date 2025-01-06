# Metrics

Democratising provides a comprehensive set of metrics for evaluating model performance, with support for custom metrics and advanced features.

## Basic Metrics

### Classification Metrics

```rust
use democratising::prelude::*;
use democratising::metrics::*;

fn classification_metrics() -> Result<()> {
    // Accuracy
    let accuracy = Accuracy::new(
        AccuracyConfig::new()
            .task(Task::MultiClass)
            .average(Average::Macro)
            .build()?,
    );
    let score = accuracy.compute(&predictions, &targets)?;

    // Precision
    let precision = Precision::new(
        PrecisionConfig::new()
            .num_classes(10)
            .average(Average::Weighted)
            .build()?,
    );
    let score = precision.compute(&predictions, &targets)?;

    // Recall
    let recall = Recall::new(
        RecallConfig::new()
            .num_classes(10)
            .average(Average::Micro)
            .build()?,
    );
    let score = recall.compute(&predictions, &targets)?;

    // F1 Score
    let f1 = F1Score::new(
        F1Config::new()
            .beta(1.0)
            .average(Average::Weighted)
            .build()?,
    );
    let score = f1.compute(&predictions, &targets)?;

    Ok(())
}
```

### Regression Metrics

```rust
fn regression_metrics() -> Result<()> {
    // Mean Squared Error
    let mse = MSE::new();
    let score = mse.compute(&predictions, &targets)?;

    // Root Mean Squared Error
    let rmse = RMSE::new();
    let score = rmse.compute(&predictions, &targets)?;

    // Mean Absolute Error
    let mae = MAE::new();
    let score = mae.compute(&predictions, &targets)?;

    // R-squared
    let r2 = R2Score::new();
    let score = r2.compute(&predictions, &targets)?;

    Ok(())
}
```

## Advanced Metrics

### Confusion Matrix

```rust
fn confusion_matrix() -> Result<()> {
    // Create confusion matrix
    let cm = ConfusionMatrix::new(
        ConfusionConfig::new()
            .num_classes(10)
            .normalize(true)
            .build()?,
    );
    
    // Compute matrix
    let matrix = cm.compute(&predictions, &targets)?;
    
    // Get specific metrics
    let true_positives = cm.true_positives()?;
    let false_positives = cm.false_positives()?;
    let precision = cm.precision()?;
    let recall = cm.recall()?;

    Ok(())
}
```

### ROC and AUC

```rust
fn roc_metrics() -> Result<()> {
    // ROC curve
    let roc = ROCCurve::new();
    let (fpr, tpr, thresholds) = roc.compute(&predictions, &targets)?;
    
    // AUC score
    let auc = AUCScore::new(
        AUCConfig::new()
            .average(Average::Macro)
            .build()?,
    );
    let score = auc.compute(&predictions, &targets)?;
    
    // PR curve
    let pr = PrecisionRecallCurve::new();
    let (precision, recall, thresholds) = pr.compute(&predictions, &targets)?;

    Ok(())
}
```

## Custom Metrics

### Creating Custom Metric

```rust
#[derive(Debug)]
struct CustomMetric {
    threshold: f64,
    average: Average,
}

impl Metric for CustomMetric {
    type Input = Tensor<f64>;
    type Output = f64;

    fn compute(&self, preds: &Self::Input, targets: &Self::Input) -> Result<Self::Output> {
        // Custom metric computation
        let pred_classes = preds.gt(self.threshold)?;
        let correct = (pred_classes == targets)?;
        
        match self.average {
            Average::Micro => correct.mean(),
            Average::Macro => {
                let class_scores = correct.mean_per_class()?;
                class_scores.mean()
            }
            Average::Weighted => {
                let weights = targets.sum_per_class()?;
                let class_scores = correct.mean_per_class()?;
                (class_scores * weights)?.mean()
            }
        }
    }
}

impl CustomMetric {
    fn new(threshold: f64, average: Average) -> Self {
        Self { threshold, average }
    }
}
```

### Using Custom Metric

```rust
fn use_custom_metric() -> Result<()> {
    let metric = CustomMetric::new(0.5, Average::Macro);
    
    // Single computation
    let score = metric.compute(&predictions, &targets)?;
    
    // With accumulation
    let mut accumulator = MetricAccumulator::new(metric);
    
    for (preds, targets) in data_loader.iter() {
        accumulator.update(&preds, &targets)?;
    }
    
    let final_score = accumulator.compute()?;

    Ok(())
}
```

## Metric Collections

### Multiple Metrics

```rust
fn multiple_metrics() -> Result<()> {
    // Create metric collection
    let metrics = MetricCollection::new(vec![
        Box::new(Accuracy::new(Default::default())),
        Box::new(Precision::new(Default::default())),
        Box::new(Recall::new(Default::default())),
    ]);

    // Compute all metrics
    let scores = metrics.compute(&predictions, &targets)?;
    
    // Access individual scores
    println!("Accuracy: {}", scores.get("accuracy")?);
    println!("Precision: {}", scores.get("precision")?);
    println!("Recall: {}", scores.get("recall")?);

    Ok(())
}
```

### Metric Accumulation

```rust
fn accumulate_metrics() -> Result<()> {
    // Create accumulator
    let mut accumulator = MetricAccumulator::new(
        MetricCollection::new(vec![
            Box::new(Accuracy::new(Default::default())),
            Box::new(F1Score::new(Default::default())),
        ]),
    );

    // Accumulate over batches
    for (preds, targets) in data_loader.iter() {
        accumulator.update(&preds, &targets)?;
    }

    // Get final scores
    let scores = accumulator.compute()?;

    Ok(())
}
```

## Best Practices

### Metric Selection

1. Choose based on task:
   ```rust
   let metric = match task {
       Task::BinaryClassification => {
           MetricCollection::new(vec![
               Box::new(Accuracy::new(Default::default())),
               Box::new(AUCScore::new(Default::default())),
           ])
       }
       Task::MultiClassClassification => {
           MetricCollection::new(vec![
               Box::new(Accuracy::new(Default::default())),
               Box::new(F1Score::new(Default::default())),
           ])
       }
       Task::Regression => {
           MetricCollection::new(vec![
               Box::new(MSE::new()),
               Box::new(R2Score::new()),
           ])
       }
   };
   ```

2. Consider class imbalance:
   ```rust
   let metric = F1Score::new(
       F1Config::new()
           .average(Average::Weighted)
           .class_weights(Some(weights))
           .build()?,
   );
   ```

### Performance Tips

1. Use accumulation for large datasets:
   ```rust
   let mut accumulator = MetricAccumulator::new(metric);
   for batch in loader.iter() {
       accumulator.update_gpu(&preds, &targets)?;
   }
   ```

2. Compute metrics on GPU:
   ```rust
   let metric = Accuracy::new(
       AccuracyConfig::new()
           .device("cuda:0")
           .build()?,
   );
   ```

## Next Steps

- Learn about [Loss Functions](loss-functions.md)
- Explore [Training](../training.md)
- Study [Evaluation](../evaluation.md)
- Understand [Model Selection](../model-selection.md)
