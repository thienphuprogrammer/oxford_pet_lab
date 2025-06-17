import tensorflow as tf
from tensorflow.keras.callbacks import *
import numpy as np
from datetime import datetime

def get_sota_callbacks(monitor='val_loss', 
                      patience=15,
                      model_name='best_model',
                      log_dir='logs',
                      reduce_lr_patience=7,
                      min_lr=1e-7,
                      factor=0.5):
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

def get_advanced_callbacks(monitor='val_loss',
                         patience=20,
                         model_name='advanced_model',
                         cosine_restart=True,
                         warmup_epochs=5):
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

def cosine_annealing_with_warmup(epoch, current_lr, warmup_epochs, total_epochs, min_lr=1e-6, max_lr=1e-3):
    """Cosine annealing với warmup phase"""
    if epoch < warmup_epochs:
        # Warmup phase
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))