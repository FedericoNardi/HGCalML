import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import uproot

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
    
# GP BIB MODEL HERE BELOW
def load_data(input_file):
    with uproot.open(input_file) as f:
        Z_vals = None
        deposits = []
        
        for wedge in range(12):
            wedge_deposits = []
            for layer in range(5):
                hname = f'h_layer_{wedge}_{layer}'
                hist = f[hname].to_hist()
                vals = hist.to_numpy()[0]
                vals = np.mean(vals, axis=1)  # Average over X-axis bins
                edges_z = hist.to_numpy()[1]
                centers_z = 0.5 * (edges_z[:-1] + edges_z[1:])
                
                if Z_vals is None:
                    Z_vals = centers_z  # Use Z values from first iteration (assumed same across wedges)
                
                wedge_deposits.append(vals)
            
            deposits.append(np.stack(wedge_deposits, axis=1))  # Shape (435, 5) for each wedge
    
    deposits = np.stack(deposits, axis=0)  # Shape (12, 435, 5)
    avg_deposits = np.mean(deposits, axis=0)  # Average over wedges, shape (435, 5)
    
    # Create coordinate pairs (x, z)
    X_coords = np.repeat(np.linspace(0, 250, 5, endpoint=True)[None, :], 435, axis=0)  # Layer indices repeated across Z bins
    Z_coords = np.tile(Z_vals[:, None], (1, 5))  # Tile Z values across layers
    
    X_data = np.c_[Z_coords.ravel(), X_coords.ravel()]  # Shape (435*5, 2)
    Y_data = avg_deposits.ravel()[:, None]/(10*10*50)#mm  # Shape (435*5, 1). Turn into a volumetric density
    return X_data, Y_data
    
def rbf_kernel(X1, X2, variance, length_scales):
    """Computes the 2D RBF kernel for (x, layer)."""
    sqdist = (
        tf.reduce_sum((X1 / length_scales) ** 2, axis=1, keepdims=True)
        - 2 * tf.matmul(X1 / length_scales, tf.transpose(X2 / length_scales))
        + tf.reduce_sum((X2 / length_scales) ** 2, axis=1)
    )
    return variance * tf.exp(-0.5 * sqdist)

class BibModel:
    def __init__(self, *args, **kwargs):
        self.input_file = '/media/disk/bib_layers/bib_e_layers_full.root'
        self.length_scales = tf.Variable([5.0, 20.0], dtype=tf.float32)  # (x, layer)
        self.variance = tf.Variable(1.0, dtype=tf.float32)
        self.noise_variance = tf.Variable(1e-6, dtype=tf.float32)

        X_data, Y_data = load_data(self.input_file)
        self.X_train = tf.convert_to_tensor(X_data, dtype=tf.float32)
        self.Y_train = tf.convert_to_tensor(Y_data, dtype=tf.float32)

        self.K_nn = rbf_kernel(self.X_train, self.X_train, self.variance, self.length_scales) + \
                    self.noise_variance * tf.eye(self.X_train.shape[0])
        self.L_nn = tf.linalg.cholesky(self.K_nn)
        self.alpha = tf.linalg.cholesky_solve(self.L_nn, self.Y_train)

        # ✅ Precompile predict once with relaxed shape
        self._predict_fn = tf.function(self._predict, experimental_relax_shapes=True)

    def _predict(self, X_test):
        K_ns = rbf_kernel(self.X_train, X_test, self.variance, self.length_scales)
        K_ss = rbf_kernel(X_test, X_test, self.variance, self.length_scales) + \
               self.noise_variance * tf.eye(X_test.shape[0])
        mu_s = tf.matmul(K_ns, self.alpha, adjoint_a=True)
        v = tf.linalg.triangular_solve(self.L_nn, K_ns, lower=True)
        var_s = K_ss - tf.matmul(v, v, transpose_a=True)
        return mu_s, tf.linalg.diag_part(var_s)

    def predict(self, X_test):
        return self._predict_fn(X_test)