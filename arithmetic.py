import numpy as np

z_samples_path = 'samples/z_20171115160208.npy'

z_samples = np.load(z_samples_path)

# smiling woman
A_indices = [100, 107, 120]

# neutral woman
B_indices = [101, 106, 13]

# neutral man
C_indices = [16, 1, 58]

mean_A = np.mean([z_samples[i] for i in A_indices], axis=0)
mean_B = np.mean([z_samples[i] for i in B_indices], axis=0)
mean_C = np.mean([z_samples[i] for i in C_indices], axis=0)

y = mean_A - mean_B + mean_C

np.save('smiling_man.npy', y)
