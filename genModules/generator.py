import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Shower:
    def __init__(self, energy, *args, **kwargs):
        self.energy = tf.convert_to_tensor(energy, dtype=tf.float32)
        # Define a meshgrid x in (-25,25) z in (0,500))
        # if spacing is not provided, use 100 points, otherwise use the provided spacing
        if 'spacing' in kwargs:
            self.dx = kwargs['spacing'][0]
            self.dy = kwargs['spacing'][1]
            self.dz = kwargs['spacing'][2]
            npoints_x = 500 // kwargs['spacing'][0]
            npoints_y = 500 // kwargs['spacing'][1]
            npoints_z = 500 // kwargs['spacing'][2]
            x = tf.linspace(-250.0, 250.0, int(npoints_x))
            y = tf.linspace(-250.0, 250.0, int(npoints_y))
            z = tf.linspace(0.0, 500.0, int(npoints_z))+0.01
        else:
            self.dx = 1.
            self.dy = 1.
            self.dz = 1.
            x = tf.linspace(-250.0, 250.0, 100) # mm; note that generator takes cm 
            y = tf.linspace(-250.0, 250.0, 100) # mm; note that generator takes cm
            z = tf.linspace(0.0, 500.0, 100)
            
        self.X, self.Y, self.Z = tf.meshgrid(x, y, z, indexing='ij')
        self.R = tf.sqrt((0.1*self.X)**2 + (0.1*self.Y)**2) # R is in cm

        # Load parameters and convert to tensors
        self._pars_a = tf.convert_to_tensor(np.loadtxt('genModules/pars/fit_pars_final_0.txt'), dtype=tf.float32)
        self._pars_b = tf.convert_to_tensor(np.loadtxt('genModules/pars/fit_pars_final_1.txt'), dtype=tf.float32)
        self._pars_c = tf.convert_to_tensor(np.loadtxt('genModules/pars/fit_pars_final_2.txt'), dtype=tf.float32)
        
        self._a, self._b, self._c = self.calculate_coefficients(self.Z, **kwargs)

        self.Z = self.Z + 1497.5 + self.dz
   
    def calculate_coefficients(self, z, DEBUG=False, *args, **kwargs):
        def func0(x, a, b, c):
            return a * (b - tf.exp(-c * x))
        
        def func1(x, a, b, c):
            return a * tf.exp(-b * x) + c
        
        def func2(x, a, b, c):
            return a * x**2 + b * x + c
            
        a0 = func0(self.energy, *self._pars_a[0])
        a1 = func1(self.energy, *self._pars_a[1])
        a2 = func0(self.energy, *self._pars_a[2])
        a3 = func0(self.energy, *self._pars_a[3])

        b0 = func1(self.energy, *self._pars_b[0])
        b1 = func1(self.energy, *self._pars_b[1])
        b2 = func1(self.energy, *self._pars_b[2])

        c0 = func1(self.energy, *self._pars_c[0])
        c1 = func1(self.energy, *self._pars_c[1])
        c2 = tf.constant(1., dtype=tf.float32)  # Use TensorFlow constant for c2

        # Plot c0, c1, c2 for a range of energies in 0.5, 150
        if DEBUG:
            print('Producing debug plots for c parameters...')
            xx = tf.linspace(0.5, 150.0, 100)
            par0 = [func1(x, *self._pars_c[0]) for x in xx]
            par1 = [func1(x, *self._pars_c[1]) for x in xx]
            par2 = [func2(x, *self._pars_c[2]) for x in xx]
            
            plt.figure(figsize=(10, 15))
            plt.subplot(3, 1, 1)
            plt.plot(xx.numpy(), np.array(par0))
            plt.title('c0(E)')
            plt.subplot(3, 1, 2)
            plt.plot(xx.numpy(), np.array(par1))
            plt.title('c1(E)')
            plt.subplot(3, 1, 3)
            plt.plot(xx.numpy(), np.array(par2))
            plt.title('c2(E)')
            plt.savefig('img/debug/c_parameters.png')
            plt.close()

        # same for b0, b1, b2
        if DEBUG:
            print('Producing debug plots for b parameters...')
            xx = tf.linspace(0.5, 150.0, 100)
            par0 = [func1(x, *self._pars_b[0]) for x in xx]
            par1 = [func1(x, *self._pars_b[1]) for x in xx]
            par2 = [func1(x, *self._pars_b[2]) for x in xx]
            
            plt.figure(figsize=(10, 15))
            plt.subplot(3, 1, 1)
            plt.plot(xx.numpy(), np.array(par0))
            plt.title('b0(E)')
            plt.subplot(3, 1, 2)
            plt.plot(xx.numpy(), np.array(par1))
            plt.title('b1(E)')
            plt.subplot(3, 1, 3)
            plt.plot(xx.numpy(), np.array(par2))
            plt.title('b2(E)')
            plt.savefig('img/debug/b_parameters.png')
            plt.close()

        # same for a0, a1, a2, a3
        if DEBUG:
            print('Producing debug plots for a parameters...')
            xx = tf.linspace(0.5, 150.0, 100)
            par0 = [func0(x, *self._pars_a[0]) for x in xx]
            par1 = [func1(x, *self._pars_a[1]) for x in xx]
            par2 = [func0(x, *self._pars_a[2]) for x in xx]
            par3 = [func0(x, *self._pars_a[3]) for x in xx]
            
            plt.figure(figsize=(10, 15))
            plt.subplot(4, 1, 1)
            plt.plot(xx.numpy(), np.array(par0))
            plt.title('a0(E)')
            plt.subplot(4, 1, 2)
            plt.plot(xx.numpy(), np.array(par1))
            plt.title('a1(E)')
            plt.subplot(4, 1, 3)
            plt.plot(xx.numpy(), np.array(par2))
            plt.title('a2(E)')
            plt.subplot(4, 1, 4)
            plt.plot(xx.numpy(), np.array(par3))
            plt.title('a3(E)')
            plt.savefig('img/debug/a_parameters.png')
            plt.close()

        _a = a0 * tf.exp(a1 * z**3 + a2 * z**2 + a3 * z)
        _b = b0 * tf.exp(-b1 * z) + b2
        _c = c0 * tf.exp(-c1 * (tf.math.log(z) - c2)**2)

        return _a, _b, _c

    def __call__(self):
        depo = tf.zeros_like(self.R)
        depo = tf.where(self.R < 25.0, tf.exp(self._a * tf.exp(-self._b * tf.abs(self.R))) + self._c - 1.0, depo)
        final =  depo*(self.dx*self.dy*self.dz)/(2*3.14159*10*10*50)
        if final.numpy().sum() > self.energy:
            return final*self.energy/final.numpy().sum()
        else:
            return final


    def plot_parameters(self):
        z = tf.linspace(0.0, 500.0, 100)
        plt.figure(figsize=(10, 15))
        a, b, c = self.calculate_coefficients(z)
        plt.subplot(3, 1, 1)
        plt.plot(z.numpy(), a.numpy())
        plt.title('a(z)')
        plt.subplot(3, 1, 2)
        plt.plot(z.numpy(), b.numpy())
        plt.title('b(z)')
        plt.subplot(3, 1, 3)
        plt.plot(z.numpy(), c.numpy())
        plt.title('c(z)')
        plt.savefig('img/parameters.png')
        plt.close()



class GravNetLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, k=10):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.feature_dim)
        ])
    
    def call(self, inputs):
        positions, features = inputs[..., :2], inputs[..., 2:]
        # Compute distance matrix
        dists = tf.norm(tf.expand_dims(positions, 2) - tf.expand_dims(positions, 1), axis=-1)
        # Find k nearest neighbors
        knn_indices = tf.argsort(dists, axis=-1)[..., 1:self.k+1]
        
        # Aggregate features from neighbors
        neighbor_features = tf.gather(features, knn_indices, batch_dims=1)
        aggregated_features = tf.reduce_mean(neighbor_features, axis=-2)
        
        # Transform features
        updated_features = self.mlp(aggregated_features)
        return tf.concat([positions, updated_features], axis=-1)

class GravNetBlock():
    '''
    A block of [GravNet, MessagePassing, BatchNormalization, Dense(128), BatchNormalization, Dense(96), GlobalExchange, Dense(96), BatchNormalization] layers
    '''
    def __init__(self, feature_dim, k=8):
        self.gravnet = GravNetLayer(feature_dim, k)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.global_exchange = self.global_exchange = tf.keras.layers.GlobalAveragePooling1D()
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
    def __call__(self, x):
        x = self.gravnet(x)
        x = self.bn1(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.dense2(x)
        x = self.global_exchange(x)
        x = self.dense3(x)
        x = self.bn3(x)
        return x
    
# Build the GNN Model
class ShowerGNN(tf.keras.Model):
    def __init__(self):
        super(ShowerGNN, self).__init__()
        
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.gravnet1 = GravNetLayer(32, k=8)
        self.gravnet2 = GravNetLayer(32, k=8)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = self.batch_norm(inputs, training=training)
        x1 = self.gravnet1(x)
        x2 = self.gravnet2(x1)
        x = tf.concat([x1, x2], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.batch_norm2(x, training=training)
        return self.output_layer(x)

        
    
    