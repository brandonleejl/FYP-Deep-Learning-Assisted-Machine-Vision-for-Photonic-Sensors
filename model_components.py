import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MCDropout(layers.Dropout):
    """
    Dropout layer that applies dropout during inference as well,
    enabling Monte Carlo Dropout for uncertainty estimation.

    Mathematical Reasoning:
    Standard Dropout is only active during training to prevent overfitting. During inference, it is turned off
    (scaled by 1/p). MC Dropout keeps dropout active during inference (training=True).
    By running multiple forward passes with different dropout masks, we obtain a distribution of predictions.
    The mean of these predictions is the final output, and the standard deviation represents the model's
    epistemic uncertainty (model uncertainty). This is crucial for safety-critical applications like pH sensing
    where we need to know how confident the model is.
    """
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def channel_attention_module(x, ratio=8):
    r"""
    Channel Attention Module (CAM).

    Mathematical Reasoning:
    Focuses on 'what' is meaningful in the input image. It exploits the inter-channel relationship of features.
    We use both Average Pooling and Max Pooling to aggregate spatial information of a feature map, generating
    two different spatial context descriptors: F_avg and F_max.
    These are then forwarded to a shared multi-layer perceptron (MLP) to produce the channel attention map.
    M_c(F) = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    This allows the network to emphasize informative channels (e.g., specific dye color channels) and suppress less useful ones.
    """
    channel = x.shape[-1]

    # Shared Dense layers (MLP)
    # The MLP consists of one hidden layer with reduction ratio r.
    shared_dense_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_dense_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    # Global Average Pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    # Global Max Pooling
    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    # Add and Sigmoid Activation
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([x, cbam_feature])

def spatial_attention_module(x):
    r"""
    Spatial Attention Module (SAM).

    Mathematical Reasoning:
    Focuses on 'where' is an informative part. It complements Channel Attention.
    It generates a spatial attention map by utilizing the inter-spatial relationship of features.
    We apply Average Pooling and Max Pooling operations along the channel axis and concatenate them to generate
    an efficient feature descriptor.
    M_s(F) = \sigma(f^{7x7}([AvgPool(F); MaxPool(F)]))
    where f^{7x7} represents a convolution operation with a filter size of 7x7.
    This helps the network to focus on the hydrogel region against the background.
    """
    # Average Pooling along channel axis
    # Result shape: (H, W, 1)
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)

    # Max Pooling along channel axis
    # Result shape: (H, W, 1)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)

    # Concatenate along channel axis
    # Result shape: (H, W, 2)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # Convolution 7x7 to produce spatial attention map
    # Result shape: (H, W, 1)
    cbam_feature = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

    return layers.Multiply()([x, cbam_feature])

def cbam_block(x, ratio=8):
    """
    Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention and then Spatial Attention.
    Ref: "CBAM: Convolutional Block Attention Module", Woo et al. (ECCV 2018).

    F' = M_c(F) * F
    F'' = M_s(F') * F'
    """
    x = channel_attention_module(x, ratio)
    x = spatial_attention_module(x)
    return x

def build_enhanced_ph_classifier(num_classes, input_shape=(224, 224, 3)):
    """
    Builds the enhanced pH classifier with:
    1. MobileNetV2 Backbone (pre-trained on ImageNet, frozen initially usually)
    2. CBAM Attention (Channel + Spatial)
    3. MC Dropout (rate=0.5) for uncertainty estimation
    4. Multi-Task Heads (Classification + Regression)
    """
    inputs = keras.Input(shape=input_shape)

    # MobileNetV2 Backbone
    try:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
        print("Classifier backbone: MobileNetV2 (ImageNet weights)")
    except Exception as e:
        print("WARNING: Could not load ImageNet weights for classifier backbone; using weights=None.")
        print("Reason:", str(e)[:200], "...")
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )

    x = backbone.output

    # Apply Attention Mechanism (CBAM) strategically after feature extraction
    x = cbam_block(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layer for feature transformation
    x = layers.Dense(128, activation="relu")(x)

    # Monte Carlo Dropout (0.5) before final output
    # This enables uncertainty estimation during inference.
    x = MCDropout(0.5)(x)

    # Multi-Task Heads

    # 1. Classification Head (predicts pH class logits)
    # Outputting logits (linear activation) for numerical stability with SparseCategoricalCrossentropy(from_logits=True)
    cls_logits = layers.Dense(num_classes, activation=None, name="classification_logits")(x)

    # 2. Regression Head (predicts continuous pH value directly)
    # Linear activation is standard for regression.
    reg_output = layers.Dense(1, activation="linear", name="regression_output")(x)

    model = keras.Model(inputs=inputs, outputs=[cls_logits, reg_output], name="enhanced_ph_classifier")
    return model
