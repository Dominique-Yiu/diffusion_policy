import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.model.our_model.discretizer import KMeansDiscretizer
import torch
import ipdb; 

# Define the parameters for the true clusters
cluster1_center = torch.Tensor([2, 2])
cluster2_center = torch.Tensor([8, 8])
cluster_std = 1.0

# Generate data points for the true clusters
num_samples_per_cluster = 50
cluster1_samples = torch.randn(num_samples_per_cluster, 2) * cluster_std + cluster1_center
cluster2_samples = torch.randn(num_samples_per_cluster, 2) * cluster_std + cluster2_center

# Combine the two clusters
true_data = torch.vstack((cluster1_samples, cluster2_samples))

discretizer = KMeansDiscretizer(feature_dim=2, num_bins=2, n_iter=100, predict_offsets=True)
discretizer.fit_discretizer(true_data)
cluster = discretizer.cluster_centers.detach()

plt.scatter(true_data[:, 0], true_data[:, 1], c='b', marker='o', label='Samples')
plt.scatter(cluster[:, 0], cluster[:, 1], c='r', marker='o', label='Clusters')

test_1 = torch.randn(3, 2) * cluster_std / 2 + cluster1_center
test_2 = torch.randn(3, 2) * cluster_std / 2 + cluster2_center


output, offset = discretizer.encod_into_latent(test_1)
print(offset)
output, offset = discretizer.encod_into_latent(test_2)
print(offset)

plt.legend()
plt.title('True Clusters and Random Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()