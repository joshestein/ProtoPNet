import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from src.protopnet import ProtoPNet


def project_prototypes(model: ProtoPNet, dataloader: DataLoader, epoch: int):
    """Project each prototype onto the nearest latent training patch for its corresponding class. See the training
    section (2.1) of the original paper."""
    model.eval()

    global_min_proto_dist = torch.full((model.num_prototypes,), torch.inf)
    global_min_fmap_patches = torch.zeros((model.num_prototypes, *model.prototype_shape[1:]))

    prototype_shape = model.prototype_shape
    num_prototypes = model.num_prototypes
    proto_height = prototype_shape[2]
    proto_width = prototype_shape[3]

    save_dir = model_dir / "images" / f"epoch-{epoch}"
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, label) in enumerate(dataloader):
        # TODO: check that image is normalised

        with torch.no_grad():
            output, proto_dist = model.project(image)

        label_to_index_map = _build_label_to_index_map(label, model)

        for prototype_index in range(num_prototypes):
            target_class = torch.argmax(model.prototype_onehot_class_representation[prototype_index]).item()

            if len(label_to_index_map[target_class]) == 0:
                continue

            proto_dist_j = proto_dist[label_to_index_map[target_class]][:, prototype_index, :, :]
            batch_min_proto_dist_j = torch.amin(proto_dist_j)

            if batch_min_proto_dist_j < global_min_proto_dist[prototype_index]:
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

                global_min_proto_dist[prototype_index] = batch_min_proto_dist_j
                global_min_fmap_patches[prototype_index] = batch_min_fmap_patch_j

    np.save(
        save_dir / f"bb-receptive-field-epoch-{epoch}.npy",
        proto_rf_boxes,
    )
    np.save(
        save_dir / f"bb-epoch-{epoch}.npy",
        proto_bound_boxes,
    )

    prototype_update = torch.reshape(global_min_fmap_patches, tuple(prototype_shape))
    # model.prototypes.data.copy_(torch.tensor(prototype_update, dtype=torch.float32))
    model.prototypes.data = prototype_update.clone().detach().requires_grad_(True)


def _build_label_to_index_map(label, model):
    label_to_index = {key: [] for key in range(model.num_output_classes)}
    for label_index, current_label in enumerate(label):
        label_to_index[current_label.item()].append(label_index)

    return label_to_index
