import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    tf = None
    _tf_import_error = e
else:
    _tf_import_error = None


def _build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(6),
        ]
    )
    return model


model = _build_model() if tf is not None else None

error_labels = {
    0: "Correct",
    1: "Arithmetic Error",
    2: "Sign Error",
    3: "Algebraic Error",
    4: "Incomplete",
    5: "Invalid"
}

def classify_error(code):
    if tf is None:
        raise ImportError(
            "TensorFlow is required for classify_error but is not installed. "
            "Install it with: pip install tensorflow"
        ) from _tf_import_error

    x = np.array([[float(code)]], dtype=np.float32)
    logits = model(x, training=False).numpy()
    return int(np.argmax(logits, axis=1)[0])
