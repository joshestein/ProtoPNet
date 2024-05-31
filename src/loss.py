import torch.nn
from torch import Tensor

from src.protopnet import ProtoPNet


class ProtoPLoss(torch.nn.Module):
    def __init__(
        self, model: ProtoPNet, cluster_cost_weight=0.8, separation_cost_weight=0.08, l1_regularization_weight=1e-4
    ):
        """The default values are sourced from the supplemental paper."""
        super().__init__()
        self.model = model
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.cluster_cost_weight = cluster_cost_weight
        self.separation_cost_weight = separation_cost_weight
        self.l1_regularization_weight = l1_regularization_weight

    def forward(self, output: Tensor, target: Tensor, min_distances: torch.Tensor):
        max_distance = self.model.prototypes.shape[1] * self.model.prototypes.shape[2] * self.model.prototypes.shape[3]

        cross_entropy = self.cross_entropy(output, target)
        cluster_cost = self._calculate_cluster_cost(
            max_distance=max_distance,
            min_distances=min_distances,
            target=target,
        )
        separation_cost = self._calculate_separation_cost(
            max_distance=max_distance,
            min_distances=min_distances,
            target=target,
        )

        # Convex optimisation of the last layer
        l1_mask = 1 - torch.transpose(self.model.prototype_onehot_class_representation, 0, 1)
        l1 = (self.model.fully_connected.weight * l1_mask).norm(p=1)

        loss = (
            cross_entropy
            + self.cluster_cost_weight * cluster_cost
            + self.separation_cost_weight * separation_cost
            + self.l1_regularization_weight * l1
        )
        return loss

    def _calculate_cluster_cost(self, max_distance: Tensor, min_distances: Tensor, target: Tensor):
        prototype_for_class = torch.t(self.model.prototype_onehot_class_representation[:, target])
        inverted_distances_to_target_prototypes, _ = torch.max(
            (max_distance - min_distances) * prototype_for_class, dim=1
        )
        return torch.mean(max_distance - inverted_distances_to_target_prototypes)

    def _calculate_separation_cost(self, max_distance: Tensor, min_distances: Tensor, target: Tensor):
        prototype_for_other_class = 1 - torch.t(self.model.prototype_onehot_class_representation[:, target])
        inverted_distances_to_nontarget_prototypes, _ = torch.max(
            (max_distance - min_distances) * prototype_for_other_class, dim=1
        )
        return torch.mean(max_distance - inverted_distances_to_nontarget_prototypes)
