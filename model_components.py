import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MCDropout(layers.Dropout):
    """
    Dropout layer that applies dropout during inference as well,
    enabling Monte Carlo Dropout for uncertainty estimation.
    """
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def channel_attention_module(x, ratio=8):
    """
    Channel Attention Module.
    """
    channel = x.shape[-1]

    # Shared Dense layers
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

    # Add and Sigmoid
    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([x, cbam_feature])

def spatial_attention_module(x):
    """
    Spatial Attention Module.
    """
    # Average Pooling along channel axis
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)

    # Max Pooling along channel axis
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)

    # Concatenate
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    # Convolution 7x7
    cbam_feature = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)

    return layers.Multiply()([x, cbam_feature])

def cbam_block(x, ratio=8):
    """
    Convolutional Block Attention Module (CBAM).
    """
    x = channel_attention_module(x, ratio)
    x = spatial_attention_module(x)
    return x

def build_enhanced_ph_classifier(num_classes, input_shape=(224, 224, 3)):
    """
    Builds the enhanced pH classifier with:
    1. MobileNetV2 Backbone
    2. CBAM Attention
    3. MC Dropout
    4. Multi-Task Heads (Classification + Regression)
    """
    inputs = keras.Input(shape=input_shape)

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

    # Apply Attention Mechanism (CBAM)
    x = cbam_block(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # MC Dropout
    x = MCDropout(0.25)(x)

    # Shared Dense Layer
    x = layers.Dense(128, activation="relu")(x)
    x = MCDropout(0.25)(x)

    # Multi-Task Heads

    # 1. Classification Head (predicts pH class probability)
    cls_output = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    # Note: original model output logits, but for multi-output it's cleaner to output probabilities or logits consistently.
    # The original code used `from_logits=True`.
    # Let's stick to logits for numerical stability if we use from_logits=True in loss,
    # BUT here I'll use explicit softmax and from_logits=False, or keep it linear.
    # The original code used linear activation for the last layer.
    # Let's keep it linear (logits) to match the original loss configuration if possible,
    # but since we are redefining the model, we can change it.
    # Wait, if I change it to softmax, I must change the loss to `from_logits=False`.
    # Let's keep it consistent with the original: Linear output (logits).
    cls_logits = layers.Dense(num_classes, activation=None, name="classification_logits")(x)

    # 2. Regression Head (predicts continuous pH value directly)
    reg_output = layers.Dense(1, activation="linear", name="regression_output")(x)

    model = keras.Model(inputs=inputs, outputs=[cls_logits, reg_output], name="enhanced_ph_classifier")
    return model
