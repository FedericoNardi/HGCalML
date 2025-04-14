# Define a class for the model to make it easier to use and import
# It should take as input any marginal distribution f_Y(y, z) and return the predicted energy deposition E(x, y, z)
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class InflateShower:
    def _compute_ring_fractions(self, grid_size=100, num_samples=1000):
        fractions = np.zeros((grid_size, grid_size, grid_size//2), dtype=np.float32)

        x_vals = np.linspace(-grid_size/2 + 0.5, grid_size/2 - 0.5, grid_size)
        y_vals = np.linspace(-grid_size/2 + 0.5, grid_size/2 - 0.5, grid_size)

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                xs = x + (np.random.rand(num_samples) - 0.5)
                ys = y + (np.random.rand(num_samples) - 0.5)
                rs = np.sqrt(xs**2 + ys**2)

                for r in range(1, grid_size//2):
                    fractions[i, j, r] = np.mean((r-1 <= rs) & (rs < r))

        return fractions

    def __init__(self, grid_size=100, num_iterations=25000):
        self.fractions = self._compute_ring_fractions(grid_size)
        self.grid_size = grid_size
        self.g_rz = tf.Variable(tf.zeros([50, grid_size]), trainable=True, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.num_iterations = num_iterations

    def model_loss(self, g_rz, fractions, I_target):
        I_pred = tf.einsum('rz,xyr->xz', g_rz, fractions)
        return tf.reduce_mean(tf.square(I_pred - I_target))

    @tf.function
    def _train_step(self, f_Y):
        with tf.GradientTape() as tape:
            loss = self.model_loss(self.g_rz, self.fractions, f_Y)
        grads = tape.gradient(loss, [self.g_rz])
        self.optimizer.apply_gradients(zip(grads, [self.g_rz]))
        self.g_rz.assign(tf.maximum(self.g_rz, 0))
        return loss

    def fit(self, f_Y):
        for i in range(self.num_iterations):
            loss_val = self._train_step(f_Y)
    
    def __call__(self, f_Y):
        f_Y = tf.where(f_Y > 1e-3, f_Y, 0)
        self.fit(f_Y)
        g_rz_pred = self.g_rz
        E_pred = tf.einsum('rz,xyr->xyz', g_rz_pred, self.fractions)
        E_pred = tf.where(E_pred > 1e-3, E_pred, 0)
        return E_pred
