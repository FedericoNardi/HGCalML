# Routine to plot training losses from .txt file

import matplotlib.pyplot as plt
import numpy as np

# Command-line input: folder containing .txt files
import sys
folder = sys.argv[1]

# Read .txt file 'losses.log'
import os
filename = os.path.join(folder, 'losses.log')

# Read .csv file import as numpy arrays
losses = np.genfromtxt(filename, delimiter=' ')
train_loss = losses[:, 0]
val_loss = losses[:, 1]

# Plot losses
plt.figure()
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# Set y limits to the range of the training loss
plt.ylim([0, 2*np.mean(train_loss)])
plt.savefig(os.path.join(folder, 'losses.png'))
