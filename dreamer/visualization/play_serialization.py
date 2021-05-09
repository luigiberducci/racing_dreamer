import tensorflow as tf
import pickle
import pathlib
from models import ActionDecoder
import tensorflow.keras.layers as tfkl

model = ActionDecoder(2, 4, 400)

for _ in range(10):
    x = tf.random.uniform([1, 230])
    model.log_activity('input', x)
    for index in range(model._layers):
        x = model.get(f'h{index}', tfkl.Dense, 100, 'relu')(x)
        model.log_activity(f'h{index}', x)
    x = model.get(f'hout', tfkl.Dense, 100, 'relu')(x)
    model.log_activity(f'output', x)
model.log_write("prova.pkl")


