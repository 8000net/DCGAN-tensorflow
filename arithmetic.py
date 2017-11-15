import numpy as np

BATCH_SIZE = 64
z_samples_path = 'samples/z_20171115160208.npy'

z_samples = np.load(z_samples_path)

# smiling woman
A_indices = [37, 100, 107]

# neutral woman
B_indices = [60, 84, 99]

# neutral man
C_indices = [58, 75, 78]

mean_A = np.mean([z_samples[i] for i in A_indices], axis=0)
mean_B = np.mean([z_samples[i] for i in B_indices], axis=0)
mean_C = np.mean([z_samples[i] for i in C_indices], axis=0)

y = mean_A - mean_B + mean_C
y_grid = np.tile(y, (BATCH_SIZE, 1))
y_grid += np.random.uniform(-0.25, 0.25, y_grid.shape)

np.save('smiling_man_grid.npy', y_grid)
