"""
Mini-batch Training Examples
============================

Các ví dụ sử dụng hệ thống mini-batching training mới.
"""

import numpy as np
import matplotlib.pyplot as plt


# ==================== Example 1: Simple Classification with MiniBatchTrainer ====================

def example_1_simple_classification():
    """Example 1: Simple classification sử dụng MiniBatchTrainer."""
    print("=" * 60)
    print("Example 1: Simple Classification with MiniBatchTrainer")
    print("=" * 60)
    
    from Model.model import (
        Model, Layer_Dense, Activation_ReLU, 
        Activation_Softmax_Loss_CategoricalCrossEntropy,
        Optimizer_Adam, Accuracy_Categorical, MiniBatchTrainer, MiniBatchConfig
    )
    
    # 1. Chuẩn bị dữ liệu
    np.random.seed(42)
    n_train, n_features, n_classes = 300, 2, 3
    
    X_train = np.random.randn(n_train, n_features) * 2
    y_train = np.random.randint(0, n_classes, n_train)
    
    X_val = np.random.randn(60, n_features) * 2
    y_val = np.random.randint(0, n_classes, 60)
    
    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation data: {X_val.shape}, Labels: {y_val.shape}")
    
    # 2. Xây dựng model
    model = Model()
    model.add(Layer_Dense(n_features, 32))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(32, 16))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(16, n_classes))
    
    # 3. Setup model
    model.set(
        loss=Activation_Softmax_Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_Adam(),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()
    
    # 4. Train với MiniBatchTrainer (mặc định)
    print("\nTraining with MiniBatchTrainer (default)...")
    history = model.train(
        X=X_train,
        y=y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        print_every=10,
        early_stopping_patience=15
    )
    
    # 5. Kết quả
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final training loss: {history['train_loss'][-1]:.5f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.5f}")
    if history['train_acc'][-1]:
        print(f"Final training accuracy: {history['train_acc'][-1]:.5f}")
    if history['val_acc'][-1]:
        print(f"Final validation accuracy: {history['val_acc'][-1]:.5f}")
    
    return model, history


# ==================== Example 2: DataLoader Usage ====================

def example_2_dataloader_usage():
    """Example 2: Tìm hiểu cách dùng DataLoader."""
    print("\n" + "=" * 60)
    print("Example 2: DataLoader Usage")
    print("=" * 60)
    
    from Model.model import DataLoader
    
    # Tạo data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    # Tạo DataLoader
    loader = DataLoader(X, y, batch_size=32, shuffle=True, random_state=42)
    
    print(f"Total samples: {len(X)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Total batches per epoch: {len(loader)}")
    
    # Iterate qua 2 epochs
    for epoch in range(2):
        print(f"\nEpoch {epoch} - Epoch count: {loader.get_epoch_count()}")
        batch_num = 0
        for X_batch, y_batch in loader:
            batch_num += 1
            print(f"  Batch {batch_num}: X shape {X_batch.shape}, y shape {y_batch.shape}")
        print(f"  Total batches in epoch: {loader.get_batch_count()}")


# ==================== Example 3: With Validation Split ====================

def example_3_validation_split():
    """Example 3: Training dengan automatic validation split."""
    print("\n" + "=" * 60)
    print("Example 3: Training with Validation Split")
    print("=" * 60)
    
    from Model.model import (
        Model, Layer_Dense, Activation_ReLU,
        Activation_Softmax_Loss_CategoricalCrossEntropy,
        Optimizer_SGD_Decay, Accuracy_Categorical
    )
    
    # Data (không chia validation trước)
    np.random.seed(42)
    X = np.random.randn(400, 10)
    y = np.random.randint(0, 2, 400)
    
    print(f"Total data: {len(X)} samples")
    
    # Model
    model = Model()
    model.add(Layer_Dense(10, 32))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(32, 2))
    
    model.set(
        loss=Activation_Softmax_Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_SGD_Decay(learning_rate=0.01),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()
    
    # Train với automatic validation split
    print("\nTraining with 20% validation split...")
    history = model.train(
        X=X,
        y=y,
        epochs=30,
        batch_size=32,
        validation_split=0.2,  # Tự động chia 20% cho validation
        print_every=10,
        early_stopping_patience=10
    )
    
    print(f"\nFinal training loss: {history['train_loss'][-1]:.5f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.5f}")


# ==================== Example 4: Simple vs MiniBatchTrainer Mode ====================

def example_4_training_modes():
    """Example 4: So sánh 2 training modes."""
    print("\n" + "=" * 60)
    print("Example 4: Training Modes Comparison")
    print("=" * 60)
    
    from Model.model import (
        Model, Layer_Dense, Activation_ReLU,
        Activation_Softmax_Loss_CategoricalCrossEntropy,
        Optimizer_Adam, Accuracy_Categorical
    )
    
    # Data
    np.random.seed(42)
    X_train = np.random.randn(200, 5)
    y_train = np.random.randint(0, 3, 200)
    X_val = np.random.randn(40, 5)
    y_val = np.random.randint(0, 3, 40)
    
    def create_model():
        model = Model()
        model.add(Layer_Dense(5, 16))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(16, 3))
        model.set(
            loss=Activation_Softmax_Loss_CategoricalCrossEntropy(),
            optimizer=Optimizer_Adam(),
            accuracy=Accuracy_Categorical()
        )
        model.finalize()
        return model
    
    # Mode 1: MiniBatchTrainer (default)
    print("\nMode 1: MiniBatchTrainer (new, recommended)")
    model1 = create_model()
    history1 = model1.train(
        X=X_train,
        y=y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        print_every=10,
        use_minibatch_trainer=True  # Explicit
    )
    
    # Mode 2: Simple loop
    print("\n\nMode 2: Simple training loop (backward compatible)")
    model2 = create_model()
    history2 = model2.train(
        X=X_train,
        y=y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        print_every=10,
        use_minibatch_trainer=False  # Old mode
    )
    
    print("\n" + "=" * 60)
    print("Both modes produced similar results!")
    print(f"MiniBatchTrainer final loss: {history1['train_loss'][-1]:.5f}")
    print(f"Simple mode final loss: {history2['train_loss'][-1]:.5f}")


# ==================== Example 5: Direct MiniBatchTrainer Usage ====================

def example_5_direct_trainer_usage():
    """Example 5: Dùng MiniBatchTrainer trực tiếp."""
    print("\n" + "=" * 60)
    print("Example 5: Direct MiniBatchTrainer Usage")
    print("=" * 60)
    
    from Model.model import (
        Model, Layer_Dense, Activation_ReLU, Activation_Linear,
        Loss_MeanSquaredError, Optimizer_Adam,
        MiniBatchTrainer, MiniBatchConfig
    )
    
    # Regression data
    np.random.seed(42)
    X_train = np.random.randn(150, 3)
    y_train = np.sum(X_train * np.array([2, -3, 1]), axis=1, keepdims=True) + np.random.randn(150, 1) * 0.5
    
    X_val = np.random.randn(30, 3)
    y_val = np.sum(X_val * np.array([2, -3, 1]), axis=1, keepdims=True) + np.random.randn(30, 1) * 0.5
    
    # Model
    model = Model()
    model.add(Layer_Dense(3, 16))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(16, 1))
    model.add(Activation_Linear())
    
    model.set(
        loss=Loss_MeanSquaredError(),
        optimizer=Optimizer_Adam(),
        accuracy=None
    )
    model.finalize()
    
    # Setup MiniBatchTrainer
    config = MiniBatchConfig(
        batch_size=32,
        shuffle=True,
        epochs=40,
        random_state=42
    )
    
    trainer = MiniBatchTrainer(model, config)
    
    # Train
    print("Training regression model...")
    history = trainer.train(
        X=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        print_every=10,
        early_stopping_patience=10
    )
    
    # Access trainer attributes
    print(f"\nBest validation loss: {trainer.best_val_loss:.5f}")
    print(f"Early stopped: {trainer.early_stop}")
    
    return model, trainer, history


# ==================== Example 6: Batch Statistics ====================

def example_6_batch_analysis():
    """Example 6: Analisis batch-level statistics."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Size Analysis")
    print("=" * 60)
    
    from Model.model import DataLoader
    
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 10, 1000)
    
    batch_sizes = [16, 32, 64, 128]
    
    for bs in batch_sizes:
        loader = DataLoader(X, y, batch_size=bs, shuffle=False, drop_last=False)
        n_batches = len(loader)
        
        # Simulate iterating
        actual_batch_count = 0
        total_samples = 0
        for X_batch, y_batch in loader:
            actual_batch_count += 1
            total_samples += len(X_batch)
        
        print(f"Batch size: {bs:3d} -> {n_batches:2d} batches, {total_samples:4d} total samples")


# ==================== Visualization ====================

def visualize_training_history(history):
    """Visualize training history."""
    plt.figure(figsize=(15, 4))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    if any(history['val_loss']):
        plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 3, 2)
    if any(history['train_acc']):
        plt.plot(history['train_acc'], label='Train Acc', marker='o')
    if any(history['val_acc']):
        plt.plot(history['val_acc'], label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')
    plt.grid(True, alpha=0.3)
    
    # Learning Rate
    plt.subplot(1, 3, 3)
    if any(history['lr']):
        plt.plot(history['lr'], marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== Main ====================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MINI-BATCH TRAINING EXAMPLES")
    print("=" * 60)
    
    # Chọn examples để chạy
    run_examples = [1, 2, 3, 4, 5, 6]
    
    try:
        if 1 in run_examples:
            model, history = example_1_simple_classification()
            # visualize_training_history(history)
        
        if 2 in run_examples:
            example_2_dataloader_usage()
        
        if 3 in run_examples:
            example_3_validation_split()
        
        if 4 in run_examples:
            example_4_training_modes()
        
        if 5 in run_examples:
            model, trainer, history = example_5_direct_trainer_usage()
            # visualize_training_history(history)
        
        if 6 in run_examples:
            example_6_batch_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
    
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
