import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from src.protopnet import ProtoPNet


def project_prototypes(model: ProtoPNet, dataloader: DataLoader):
    """Project each prototype onto the nearest latent training patch for its corresponding class. See the training
    section (2.1) of the original paper."""
    global_min_proto_dist = np.full(model.num_prototypes, np.inf)
    global_min_fmap_patches = np.zeros((model.num_prototypes, *model.prototype_shape[1:]))

    prototype_shape = model.prototype_shape
    num_prototypes = model.num_prototypes
    proto_height = prototype_shape[2]
    proto_width = prototype_shape[3]

    for i, (image, label) in enumerate(dataloader):
        # TODO: check that image is normalised

        with torch.no_grad():
            output, proto_dist = model.project(image)

        label_to_index_map = _build_label_to_index_map(label, model)

        for j in range(num_prototypes):
            target_class = torch.argmax(model.prototype_onehot_class_representation[j])

            # if len(label_to_index_map[target_class]) == 0:
            #     continue
            #
            proto_dist_j = proto_dist[label_to_index_map[target_class]][:, j, :, :]
            batch_min_proto_dist_j = torch.amin(proto_dist_j)

            if batch_min_proto_dist_j < global_min_proto_dist[j]:
                batch_argmin_proto_dist_j = list(
                    np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
                )
                batch_argmin_proto_dist_j[0] = label_to_index_map[target_class][batch_argmin_proto_dist_j[0]]

                img_index_in_batch = batch_argmin_proto_dist_j[0]
                fmap_height_start_index = batch_argmin_proto_dist_j[1]
                fmap_height_end_index = fmap_height_start_index + proto_height
                fmap_width_start_index = batch_argmin_proto_dist_j[2]
                fmap_width_end_index = fmap_width_start_index + proto_width

                batch_min_fmap_patch_j = output[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index,
                ]

                global_min_proto_dist[j] = batch_min_proto_dist_j
                global_min_fmap_patches[j] = batch_min_fmap_patch_j


def _build_label_to_index_map(label, model):
    label_to_index = {key: [] for key in range(model.num_output_classes)}
    for label_index, current_label in enumerate(label):
        label_to_index[current_label.item()].append(label_index)

    return label_to_index
