from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

try:
    import segmentation_models as sm
    sm.set_framework('tf.keras')
except ImportError:
    sm = None

# ===================== ATTENTION & FEATURE ENHANCEMENT MODULES =====================

class SpatialAttention(layers.Layer):
    """Spatial Attention Module for feature enhancement"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = layers.Concatenate()([avg_pool, max_pool])
        attention = self.conv(concat)
        return inputs * attention

class ChannelAttention(layers.Layer):
    """Channel Attention Module (SE-Block style)"""
    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.dense1 = layers.Dense(self.channels // self.reduction, activation='relu')
        self.dense2 = layers.Dense(self.channels, activation='sigmoid')
        super().build(input_shape)
        
    def call(self, inputs):
        # Global Average Pooling
        gap = layers.GlobalAveragePooling2D()(inputs)
        # Global Max Pooling
        gmp = layers.GlobalMaxPooling2D()(inputs)
        
        # Channel attention for both
        gap_attention = self.dense2(self.dense1(gap))
        gmp_attention = self.dense2(self.dense1(gmp))
        
        attention = gap_attention + gmp_attention
        attention = tf.expand_dims(tf.expand_dims(attention, 1), 1)
        
        return inputs * attention

class CBAM(layers.Layer):
    """Convolutional Block Attention Module"""
    def __init__(self, reduction=16, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

class PPM(layers.Layer):
    """Pyramid Pooling Module"""
    def __init__(self, pool_sizes=[1, 2, 3, 6], **kwargs):
        super().__init__(**kwargs)
        self.pool_sizes = pool_sizes
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.convs = []
        for _ in self.pool_sizes:
            self.convs.append(layers.Conv2D(self.channels // 4, 1, activation='relu'))
        super().build(input_shape)
        
    def call(self, inputs):
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        pool_outs = [inputs]
        
        for i, pool_size in enumerate(self.pool_sizes):
            pooled = layers.AveragePooling2D(pool_size, strides=pool_size)(inputs)
            pooled = self.convs[i](pooled)
            pooled = tf.image.resize(pooled, [h, w], method='bilinear')
            pool_outs.append(pooled)
            
        return layers.Concatenate()(pool_outs)

# ===================== ENHANCED BUILDING BLOCKS =====================

class ResidualBlock(layers.Layer):
    """Enhanced Residual Block with attention"""
    def __init__(self, filters, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_attention = use_attention
        
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        if use_attention:
            self.attention = CBAM()
            
        self.shortcut_conv = None
        
    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_conv = layers.Conv2D(self.filters, 1, padding='same')
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        shortcut = inputs
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(shortcut)
            
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.use_attention:
            x = self.attention(x)
            
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        
        return x

class DenseBlock(layers.Layer):
    """Dense Block for feature reuse"""
    def __init__(self, growth_rate=32, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.convs = []
        
        for _ in range(num_layers):
            self.convs.append(layers.Conv2D(growth_rate, 3, padding='same', activation='relu'))
            
    def call(self, inputs, training=None):
        features = [inputs]
        
        for conv in self.convs:
            x = layers.Concatenate()(features)
            x = layers.BatchNormalization()(x, training=training)
            x = conv(x)
            x = layers.Dropout(0.1)(x, training=training)
            features.append(x)
            
        return layers.Concatenate()(features)
