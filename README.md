# ISL Transfer Learning Model - EfficientNetB0

## Overview
This project implements a Transfer Learning approach for Indian Sign Language (ISL) classification using EfficientNetB0 as the backbone architecture. The model is optimized for training on Kaggle and includes both frozen backbone training and fine-tuning phases.

## Model Architecture

### Base Model
- **Backbone**: EfficientNetB0 (pre-trained on ImageNet)
- **Input Shape**: 224 x 224 x 3
- **Pooling**: Global Average Pooling 2D

### Custom Classification Head
- Dense layer (256 units, ReLU activation)
- Dropout (0.5)
- Output layer (Softmax activation, num_classes)

## Dataset Configuration

### Input Data
- **Path**: `/kaggle/input/data-isl`
- **Format**: Directory structure with class-based folders
- **Image Size**: 224 x 224 pixels
- **Batch Size**: 32

### Data Split
- Training: 80%
- Validation: 20%

## Data Augmentation

The following augmentation techniques are applied during training:
- Rotation (range: 15 degrees)
- Width shift (10%)
- Height shift (10%)
- Shear transformation (10%)
- Zoom (10%)
- Brightness adjustment (0.9 to 1.1)
- EfficientNet-specific preprocessing

## Training Strategy

### Phase 1: Frozen Backbone Training
- **Epochs**: 20
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Base Model**: Frozen (trainable=False)

### Phase 2: Fine-Tuning
- **Epochs**: 20
- **Learning Rate**: 1e-5
- **Fine-tuned Layers**: Last 50 layers of EfficientNetB0
- **Remaining Layers**: Frozen

## Class Balancing

The model uses computed class weights to handle imbalanced datasets. Weights are calculated using scikit-learn's `compute_class_weight` function with the "balanced" strategy.

## Callbacks

### Early Stopping
- **Monitor**: Validation loss
- **Patience**: 6 epochs
- **Restore Best Weights**: True

### Learning Rate Reduction
- **Monitor**: Validation loss
- **Factor**: 0.5
- **Patience**: 3 epochs
- **Minimum Learning Rate**: 1e-6

### Model Checkpoint
- **Monitor**: Validation accuracy
- **Save Best Only**: True
- **Output Path**: `/kaggle/working/isl_efficientnet_best.keras`

## Output Files

### Saved Models
1. **Best Model**: `/kaggle/working/isl_efficientnet_best.keras`
   - Saved automatically during training when validation accuracy improves
   
2. **Final Model**: `/kaggle/working/isl_efficientnet_final.keras`
   - Saved after completing both training phases

### Training Visualizations
- Accuracy plot (Training vs Validation)
- Loss plot (Training vs Validation)

## Requirements

### Dependencies
```
tensorflow>=2.x
numpy
matplotlib
scikit-learn
```

### Hardware Recommendations
- GPU-enabled environment (Kaggle GPU accelerator recommended)
- Minimum 12GB RAM
- GPU memory: 8GB+ recommended

## Usage Instructions

### Running on Kaggle

1. **Upload Dataset**
   - Upload ISL dataset to Kaggle
   - Ensure dataset path is `/kaggle/input/data-isl`
   - Dataset should be organized in class-based subdirectories

2. **Enable GPU**
   - Go to Notebook settings
   - Select "GPU" as accelerator

3. **Run the Script**
   - Execute all cells sequentially
   - Monitor training progress in output

4. **Download Models**
   - After training completes, download models from output folder
   - Available in `/kaggle/working/` directory

### Expected Directory Structure
```
/kaggle/input/data-isl/
    class_1/
        image1.jpg
        image2.jpg
        ...
    class_2/
        image1.jpg
        image2.jpg
        ...
    ...
```

## Model Performance Metrics

The model tracks the following metrics during training:
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

Performance visualizations are automatically generated after each training phase.

## Key Features

1. **Transfer Learning**: Leverages pre-trained EfficientNetB0 weights for faster convergence
2. **Two-Phase Training**: Frozen backbone training followed by fine-tuning
3. **Class Balancing**: Automatic computation of class weights for imbalanced datasets
4. **Advanced Callbacks**: Early stopping, learning rate scheduling, and model checkpointing
5. **Data Augmentation**: Comprehensive augmentation pipeline for improved generalization
6. **Kaggle Optimized**: Configured for seamless execution on Kaggle platform

## Customization Options

### Adjusting Hyperparameters
- Modify `batch_size` for memory optimization
- Change `epochs` for longer/shorter training
- Adjust learning rates (`1e-4` and `1e-5`) for convergence tuning
- Modify dropout rate (currently `0.5`) for regularization

### Fine-Tuning Configuration
- Change number of layers to fine-tune (currently last 50 layers)
- Adjust fine-tuning learning rate for stability

### Data Augmentation
- Modify augmentation parameters in `ImageDataGenerator`
- Add or remove augmentation techniques as needed

## Troubleshooting

### Common Issues

**Out of Memory Error**
- Reduce batch size
- Decrease image size
- Reduce number of fine-tuned layers

**Slow Training**
- Ensure GPU is enabled
- Increase batch size (if memory allows)
- Reduce data augmentation complexity

**Poor Performance**
- Increase training epochs
- Adjust learning rates
- Modify class weights
- Enhance data augmentation

## Model Deployment

The saved model can be loaded and used for inference:

```python
from tensorflow.keras.models import load_model

model = load_model('/kaggle/working/isl_efficientnet_final.keras')
```

## License

Ensure compliance with EfficientNet and ImageNet licensing requirements when using this model in production environments.

## Acknowledgments

- EfficientNet architecture by Google Research
- ImageNet dataset for pre-trained weights
- Kaggle platform for computational resources
