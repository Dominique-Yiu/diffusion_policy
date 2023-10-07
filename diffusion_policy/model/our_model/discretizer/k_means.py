import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import ipdb
import logging
from typing import Optional, List, Tuple
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


logger = logging.getLogger(__name__)
class KMeansDiscretizer(DictOfTensorMixin):
    def __init__(self,
                 feature_dim: int,
                 num_bins: int,
                 n_iter: int,
                 predict_offsets: bool=False) -> None:
        """Parameters:
        feature_dim: image feature shape
        num_bins: number of classes
        predict_offsets: whether output offsets
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.n_iter = n_iter
        self.predict_offsets = predict_offsets
        self.bin_centers = None

    def fit_discretizer(self, 
                        input_features: torch.Tensor) -> None:
        assert(
            self.feature_dim == input_features.shape[-1]
        ), f"Input images dimension {self.feature_dim} does not fitted model {input_features.shape[-1]}"
        
        flattened_features = input_features.view(-1, self.feature_dim)
        clusters = KMeansDiscretizer._kmeans(
            sample = flattened_features,
            ncluster = self.num_bins,
            niter = self.n_iter
        )
        self.bin_centers = clusters.detach()
        logger.info(
            f"bin_centers: {self.bin_centers}"
        )

    @classmethod
    def _kmeans(cls,
                sample: torch.Tensor,
                ncluster: int,
                niter: int):
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT library
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = sample.shape
        # randomly select clusters
        clusters = sample[torch.randperm(N)[:ncluster]]

        pbar = tqdm.trange(niter, leave=False)
        pbar.set_description("K-means clustering")

        for i in pbar:
            # assign all pixels to the closest codebook element
            idx = ((sample[:, None, :] - clusters[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to the mean of the pixels that assigned to it
            clusters = torch.stack([sample[idx == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(clusters), dim=-1)
            ndead = nanix.sum().item()

            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            clusters[nanix] = sample[torch.randperm(N)[:ndead]]
        pbar.close()
        
        return clusters
    
    @property
    def cluster_centers(self):
        return self.bin_centers
    @property
    def discretized_space(self):
        return self.num_bins

    def encode_into_latent(self,
                          input_features: torch.Tensor,
                          input_rep: Optional[torch.Tensor]=None):
        """
        Given the input image features, discretize it using the k-means clustering algorithm.

        Input:
            input_features (shape: ... x feature_dim): The input features to discretize.
        Output:
            discretized_features (shape: ... x num_tokens): The discretized features.
        """
        assert (
            self.feature_dim == input_features.shape[-1]
        ), f'Input features dimension {input_features.shape[-1]} does not match fitted model {self.feature_dim}'

        # flatten the input features
        flattened_features = input_features.view(-1, self.feature_dim)

        # obtain teh closest cluster
        closest_cluster_index = torch.argmin(
            torch.sum(
                (flattened_features[:, None, :] - self.bin_centers[None, :, :]) ** 2,
                dim = 2,
            ),
            dim = 1,
        )

        # reshape to the original shape
        discretized_cluster_index  = closest_cluster_index.view(input_features.shape[:-1] + (1,))

        if self.predict_offsets:
            # decode from laten and get the differenc
            reconstruct_features = self.decode_from_latent(discretized_cluster_index)
            offsets = input_features - reconstruct_features
            return discretized_cluster_index, offsets
        else:
            return closest_cluster_index
        
    
    def decode_from_latent(self,
                           latent_features: torch.Tensor,
                           input_rep: Optional[torch.Tensor] = None):
        offsets = None
        if type(latent_features) == tuple:
            latent_features, offsets = latent_features
        closest_cluster_center = self.bin_centers.data[latent_features]
        reconstructed_festures = closest_cluster_center.view(
            latent_features.shape[:-1] + (self.feature_dim,)
        )

        if offsets is not None:
            reconstructed_festures += offsets
        
        return reconstructed_festures