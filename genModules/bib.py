import tensorflow.keras as keras
import numpy as np
import tensorflow as tf

# Define the model
class Model:
    def __init__(self):
        self.model = self.build_model()
        # Load weights
        self.model.load_weights('genModels/BIB_model.h5')
    
    def build_model(self):
        # use functional API to build the model
        inputs = keras.Input(shape=(2,))
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        return model
    
    def predict(self, x):
        return self.model.predict(x, ) /10594.843344 # Return volumetric density

