import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Create output directory for frames
output_dir = "particle_frames"
os.makedirs(output_dir, exist_ok=True)

class StochasticBirthSimulation(tf.keras.Model):
    def __init__(self, V, T):
        super().__init__()
        self.V = tf.constant(V, dtype=tf.float32)  # Box size
        self.T = tf.constant(T, dtype=tf.float32)  # Temperature
        
        self.birth_logits = tf.Variable(tf.random.uniform([50], -3, -1), dtype=tf.float32)
        self.particle_positions = tf.Variable(tf.random.uniform([50, 2], 0, V), dtype=tf.float32)

    def sample_births(self, tau=1.0):
        """Stable Gumbel-Softmax sampling."""
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(self.birth_logits), 0.01, 1.0)))
        gumbel_logits = (self.birth_logits + gumbel_noise) / tau
        birth_probs = tf.nn.sigmoid(gumbel_logits)
        return birth_probs

    def active_particles(self):
        """Select active particles."""
        birth_probs = self.sample_births()
        return self.particle_positions * tf.expand_dims(birth_probs, axis=-1)

    def pairwise_distance(self, positions):
        """Compute pairwise distances."""
        diff = tf.expand_dims(positions, 1) - tf.expand_dims(positions, 0)
        distances = tf.norm(diff, axis=-1) + tf.eye(tf.shape(positions)[0]) * 1e6
        return distances

    def lennard_jones_potential(self, r, epsilon=1.0, sigma=1.0):
        """Stabilized Lennard-Jones potential."""
        r_safe = tf.maximum(r, 0.1)
        r6 = (sigma / r_safe) ** 6
        r12 = r6 ** 2
        return 4 * epsilon * (r12 - r6)

    def interaction_energy(self):
        """Compute total energy."""
        active_pos = self.active_particles()
        distances = self.pairwise_distance(active_pos)
        energy = tf.reduce_sum(self.lennard_jones_potential(distances))
        return energy

    def call(self, inputs=None):
        """Compute energy with regularized N."""
        N_soft = tf.reduce_sum(self.sample_births())
        energy = self.interaction_energy()
        return 0.1*energy + (N_soft - 10.0) ** 2

# Instantiate model
model = StochasticBirthSimulation(V=10.0, T=50.0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

def save_frame(step):
    """Save a frame of active particles."""
    active_positions = model.active_particles().numpy()
    plt.figure(figsize=(5, 5))
    plt.xlim(0, model.V.numpy())
    plt.ylim(0, model.V.numpy())
    plt.scatter(active_positions[:, 0], active_positions[:, 1], c='blue', alpha=0.7)
    plt.title(f"Step {step}")
    plt.savefig(f"{output_dir}/frame_{step:03d}.png")
    plt.close()

@tf.function
def train_step():
    """Perform one training step."""
    with tf.GradientTape() as tape:
        loss = model(None)
    grads = tape.gradient(loss, model.trainable_variables)
    # grads, _ = tf.clip_by_global_norm(grads, 10.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop in epochs and batches. Save frames every 10 epochs. Monitor plots for loss and active particles.
n_burn_in = 150
n_epochs = 1000
n_batches = 64

total_loss = []
active_n = []

for epoch in tqdm(range(n_epochs+n_burn_in), desc="Training"):
    batch_loss = []
    active_n_batch = []
    for batch in range(n_batches):
        loss = train_step()
        batch_loss.append(loss)
        active_n_batch.append(tf.reduce_sum(model.sample_births()).numpy())
    total_loss.append(np.mean(batch_loss))
    active_n.append(np.mean(active_n_batch))
    if epoch>n_burn_in: # factor in some burn-in
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(total_loss[n_burn_in:])
        plt.ylim(0, np.mean(total_loss[epoch-10:])*1.5)
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(active_n[n_burn_in:])
        plt.title("Active Particles")
        plt.savefig("img/landau_training.png")
        plt.close()
    if epoch % 10 == 0:
        #print(f"Epoch {epoch}: Loss = {loss.numpy()}")
        save_frame(epoch)

# Generate GIF
frames = [imageio.imread(f"{output_dir}/frame_{i:03d}.png") for i in range(0, 1000, 10)]
imageio.mimsave("particle_evolution.gif", frames, duration=0.2)
print("GIF saved as 'particle_evolution.gif'.")
