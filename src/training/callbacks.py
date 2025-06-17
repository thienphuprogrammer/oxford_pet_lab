import tensorflow as tf
from tensorflow.keras.callbacks import *
import numpy as np
from datetime import datetime

def get_sota_callbacks(
    monitor: str = 'val_loss', 
    patience: int = 15,
    model_name: str = 'best_model',
    log_dir: str = 'logs',
    reduce_lr_patience: int = 7,
    min_lr: float = 1e-7,
    factor: float = 0.5
):
    """
    Tạo bộ callbacks SOTA sử dụng built-in callbacks của Keras
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # 1. Early Stopping - Dừng training khi không cải thiện
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='auto'
        ),
        
        # 2. Model Checkpoint - Save best model
        ModelCheckpoint(
            filepath=f'{model_name}_{timestamp}.h5',
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='auto'
        ),
        
        # 3. Reduce LR on Plateau - Giảm learning rate khi plateau
        ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='auto',
            cooldown=0
        ),
        
        # 4. TensorBoard - Visualization
        TensorBoard(
            log_dir=f'{log_dir}/{model_name}_{timestamp}',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2,  # Profile batch 2 để analyze performance
            embeddings_freq=1
        ),
        
        # 5. CSV Logger - Log metrics to CSV
        CSVLogger(
            filename=f'{model_name}_{timestamp}_log.csv',
            separator=',',
            append=False
        ),
        
        # 6. Terminate on NaN - Dừng khi có NaN
        TerminateOnNaN(),
        
        # 7. Progress Bar với tqdm (nếu cần)
        # TqdmCallback(verbose=2)  # Uncomment nếu có tqdm
    ]
    
    return callbacks

def get_advanced_callbacks(
    monitor: str = 'val_loss',
    patience: int = 20,
    model_name: str = 'advanced_model',
    cosine_restart: bool = True,
    warmup_epochs: int = 5
):
    """
    Callbacks nâng cao với Cosine Annealing và Warmup
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Basic callbacks
        EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(f'{model_name}_{timestamp}.h5', monitor=monitor, save_best_only=True, verbose=1),
        TensorBoard(f'logs/{model_name}_{timestamp}', histogram_freq=1, write_graph=True),
        CSVLogger(f'{model_name}_{timestamp}_log.csv'),
        TerminateOnNaN(),
        
        # Advanced LR scheduling
        LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_with_warmup(epoch, lr, warmup_epochs, 100),
            verbose=1
        ) if cosine_restart else ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=7, verbose=1),
        
        # Backup checkpoints mỗi 10 epochs
        ModelCheckpoint(
            filepath=f'checkpoints/{model_name}_epoch_{timestamp}.h5',
            save_freq=10,  # Save every 10 epochs
            verbose=0
        ),
    ]
    
    return callbacks

def cosine_annealing_with_warmup(
    epoch: int,
    current_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
    max_lr: float = 1e-3
):
    """Cosine annealing với warmup phase"""
    if epoch < warmup_epochs:
        # Warmup phase
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

# =====================================================
# CÁCH SỬ DỤNG
# =====================================================

# 1. Setup cơ bản - đơn giản nhất
def basic_setup(
    monitor: str = 'val_accuracy',
    patience: int = 15,
    model_name: str = 'my_model'
):
    callbacks = get_sota_callbacks(
        monitor=monitor,  # hoặc 'val_loss'
        patience=patience,
        model_name=model_name
    )
    return callbacks

# 2. Setup nâng cao
def advanced_setup(
    monitor: str = 'val_accuracy',
    patience: int = 20,
    model_name: str = 'advanced_model',
    cosine_restart: bool = True,
    warmup_epochs: int = 10
):
    callbacks = get_advanced_callbacks(
        monitor=monitor,
        patience=patience,
        model_name=model_name,
        cosine_restart=cosine_restart,
        warmup_epochs=warmup_epochs
    )
    return callbacks

# 3. Custom mix - tự chọn callbacks
def custom_setup(
    monitor: str = 'val_loss',
    patience: int = 10,
    model_name: str = 'my_model'
):
    return [
        EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        ModelCheckpoint(f'{model_name}.h5', monitor=monitor, save_best_only=True),
        ReduceLROnPlateau(monitor=monitor, factor=0.2, patience=5, min_lr=1e-7),
        TensorBoard(f'logs/{model_name}', histogram_freq=1),
    ]

