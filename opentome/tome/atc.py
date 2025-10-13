# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ACT: https://github.com/JoakimHaurum/ATC
# --------------------------------------------------------

# ------ jinxin modified ------ #
import torch
import numpy as np
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from .tome import parse_r


def agglomerative_clustering(
    metric: torch.Tensor,
    num_clusters: int,
    linkage: str = "average",
    class_token: bool = False,
    distill_token: bool = False,
):
    """
    Input size is [batch, tokens, channels].
    num_clusters indicates the number of clusters to construct 
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    B, T, _ = metric.shape

    num_clusters = min(num_clusters, T-protected)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        scores = metric @ metric.transpose(-1, -2)

        if class_token:
            scores = scores[:, 1:, 1:]
            T -= 1

        #upper_traingle_indexes = np.triu_indices(T, k=1)
        scores = (1 - scores).cpu().numpy()
        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="precomputed", 
                                             linkage=linkage, distance_threshold=None
                                            )

        cluster_labels = np.zeros((B, T),dtype=np.int64)
        for b_idx in range(B):
            labels = clustering.fit(scores[b_idx]).labels_
            cluster_labels[b_idx] = labels
            
        cluster_labels = torch.from_numpy(cluster_labels).to(device = metric.device)

        if class_token:
            # Sort to ensure the class token is at the start
            cluster_labels = cluster_labels + protected
            cluster_labels = torch.cat([torch.zeros(B, 1, device = metric.device).long(), cluster_labels], dim=-1)

   
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        C = x.shape[-1]
        dst  = torch.zeros(B, num_clusters + protected, C, device=x.device)
        dst = dst.scatter_reduce(-2, cluster_labels.unsqueeze(-1).repeat(1, 1, C), x, reduce=mode)
        return dst
    

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # TODO: This method do not support unmerge
        pass
    
    return merge, unmerge, cluster_labels



def atc_parse_r(
    num_layers: int, r: Union[List[int], Tuple[int, float], int], total: int = None, \
    offcial: bool = True, location: list = [], ratio: float = 0.5,
) -> List[int]:
    if not offcial:
        return parse_r(num_layers, r, total)
    else:
        r_list = []
        current_tokens = total
        for i in range(num_layers):
            if i in location:
                merge_token = int(current_tokens * ratio)
                current_tokens = current_tokens - merge_token
            else:
                merge_token = 0
            r_list.append(merge_token)
        return r_list