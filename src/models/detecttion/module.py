import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class AttentionBlock(layers.Layer):
    """Self-attention mechanism for feature enhancement"""
    
    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(self.channels // self.reduction, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.channels))
        super().build(input_shape)
    
    def call(self, inputs):
        # Channel attention
        attention = self.global_pool(inputs)
        attention = self.fc1(attention)
        attention = self.fc2(attention)
        attention = self.reshape(attention)
        
        # Apply attention
        return inputs * attention

class FPN(layers.Layer):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, feature_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        # Lateral connections
        self.lateral_convs = []
        self.output_convs = []
        
        num_levels = len(input_shape)
        for i in range(num_levels):
            lateral_conv = layers.Conv2D(
                self.feature_dim, 1, padding='same',
                name=f'lateral_conv_{i}'
            )
            output_conv = layers.Conv2D(
                self.feature_dim, 3, padding='same',
                name=f'output_conv_{i}'
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs should be a list of feature maps from different scales
        # From high resolution to low resolution
        
        # Start from the highest level feature
        laterals = []
        for i, feature in enumerate(inputs):
            lateral = self.lateral_convs[i](feature)
            laterals.append(lateral)
        
        # Top-down pathway
        fpn_features = []
        prev_feature = laterals[-1]  # Start from lowest resolution
        fpn_features.append(self.output_convs[-1](prev_feature))
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous feature
            upsampled = self.upsample(prev_feature)
            
            # Add with lateral connection
            # Ensure spatial dimensions match
            target_shape = tf.shape(laterals[i])
            upsampled = tf.image.resize(upsampled, [target_shape[1], target_shape[2]])
            
            merged = laterals[i] + upsampled
            fpn_feature = self.output_convs[i](merged)
            fpn_features.insert(0, fpn_feature)
            prev_feature = merged
        
        return fpn_features


class BiFPN(layers.Layer):
            def __init__(self, feature_dim=64, **kwargs):
                super().__init__(**kwargs)
                self.feature_dim = feature_dim
                
            def build(self, input_shape):
                num_levels = len(input_shape)
                
                # Resizing convs to match feature dimensions
                self.resample_convs = []
                for i in range(num_levels):
                    conv = layers.Conv2D(self.feature_dim, 1, padding='same', name=f'resample_{i}')
                    self.resample_convs.append(conv)
                
                # BiFPN convs
                self.bifpn_convs = []
                for i in range(num_levels):
                    conv = layers.Conv2D(self.feature_dim, 3, padding='same', name=f'bifpn_{i}')
                    self.bifpn_convs.append(conv)
                
                super().build(input_shape)
            
            def call(self, inputs):
                # Resample all inputs to same channel dimension
                resampled = []
                for i, inp in enumerate(inputs):
                    resampled.append(self.resample_convs[i](inp))
                
                # Top-down path
                top_down = []
                prev = resampled[-1]  # Start from lowest resolution
                top_down.append(prev)
                
                for i in range(len(resampled) - 2, -1, -1):
                    # Upsample and add
                    target_shape = tf.shape(resampled[i])[1:3]
                    upsampled = tf.image.resize(prev, target_shape)
                    upsampled = tf.cast(upsampled, resampled[i].dtype)
                    merged = resampled[i] + upsampled
                    top_down.insert(0, merged)
                    prev = merged
                
                # Bottom-up path
                bottom_up = []
                prev = top_down[0]  # Start from highest resolution
                bottom_up.append(self.bifpn_convs[0](prev))
                
                for i in range(1, len(top_down)):
                    # Downsample and add
                    target_shape = tf.shape(top_down[i])[1:3]
                    downsampled = layers.MaxPooling2D(pool_size=2)(prev)
                    downsampled = tf.image.resize(downsampled, target_shape)
                    downsampled = tf.cast(downsampled, top_down[i].dtype)
                    merged = top_down[i] + downsampled
                    bottom_up.append(self.bifpn_convs[i](merged))
                    prev = merged
                
                return bottom_up


# src/models/detection/heads.py
import tensorflow as tf
from tensorflow.keras import layers

class DetectionHead(layers.Layer):
    def __init__(self, n_conv=3, f=256, **kw):
        super().__init__(**kw)
        self.convs = [layers.Conv2D(f, 3, padding='same', activation='swish')
                      for _ in range(n_conv)]
        self.out_conv = layers.Conv2D(4, 3, padding='same', name='bbox_conv')
        self.gap = layers.GlobalAveragePooling2D()
        self.avg = layers.Average()

    def call(self, feats):
        outs = []
        for x in feats:
            for conv in self.convs:
                x = conv(x)
            outs.append(self.gap(self.out_conv(x)))
        return self.avg(outs)          # (B,4)

class ClassificationHead(layers.Layer):
    def __init__(self, num_classes, n_conv=3, f=256, **kw):
        super().__init__(**kw)
        self.convs = [layers.Conv2D(f, 3, padding='same', activation='swish')
                      for _ in range(n_conv)]
        self.out_conv = layers.Conv2D(num_classes, 3, padding='same')
        self.gap = layers.GlobalAveragePooling2D()
        self.avg = layers.Average()
        self.softmax = layers.Activation('softmax')

    def call(self, feats):
        outs = []
        for x in feats:
            for conv in self.convs:
                x = conv(x)
            outs.append(self.gap(self.out_conv(x)))
        return self.softmax(self.avg(outs))   # (B,num_classes)
