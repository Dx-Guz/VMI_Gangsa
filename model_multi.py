# model.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional,
    Dense, Dropout, Concatenate, Layer
)
from tensorflow.keras.models import Model

class HandSplitter(Layer):
    """Keras 3 safe alternative to Lambda for feature slicing"""
    def call(self, inputs):
        # inputs shape: (batch, seq, 128)
        right = inputs[:, :, :63]   # Right hand landmarks
        left  = inputs[:, :, 63:126]# Left hand landmarks
        bilah = inputs[:, :, 126:]  # Positional context
        return right, left, bilah

    def get_config(self):
        return super().get_config()

def build_model(
    sequence_length=15,
    feature_dim=128,
    num_classes=3,
    lstm_units_1=64,
    lstm_units_2=32,
    dropout_rate=0.3
):
    inputs = Input(shape=(sequence_length, feature_dim), name="input_sequence")
    
    # Keras 3 safe split
    right, left, bilah = HandSplitter(name="feature_splitter")(inputs)

    # RIGHT STREAM
    r = Bidirectional(LSTM(lstm_units_1, return_sequences=True), name="right_bilstm_1")(right)
    r = Dropout(dropout_rate, name="drop_r_1")(r)
    r = Bidirectional(LSTM(lstm_units_2), name="right_bilstm_2")(r)

    # LEFT STREAM
    l = Bidirectional(LSTM(lstm_units_1, return_sequences=True), name="left_bilstm_1")(left)
    l = Dropout(dropout_rate, name="drop_l_1")(l)
    l = Bidirectional(LSTM(lstm_units_2), name="left_bilstm_2")(l)

    # BILAH STREAM
    b = Bidirectional(LSTM(16), name="bilah_bilstm")(bilah)

    # RIGHT HEAD
    x_r = Concatenate(name="fusion_right")([r, b])
    x_r = Dense(64, activation='relu', name="dense_r_1")(x_r)
    x_r = Dropout(dropout_rate, name="drop_r_2")(x_r)
    out_right = Dense(num_classes, activation='softmax', name='right_output')(x_r)

    # LEFT HEAD
    x_l = Concatenate(name="fusion_left")([l, b])
    x_l = Dense(64, activation='relu', name="dense_l_1")(x_l)
    x_l = Dropout(dropout_rate, name="drop_l_2")(x_l)
    out_left = Dense(num_classes, activation='softmax', name='left_output')(x_l)

    # MODEL
    model = Model(inputs=inputs, outputs=[out_right, out_left], name="Gangsa_BiLSTM_Multi")

    # 🔑 Keras 3 strict compliance: LIST format + loss_weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=['categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights=[1.0, 1.0],
        metrics=['accuracy', 'accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()