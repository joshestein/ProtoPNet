import torch
import torch.nn.functional as F
from torch import nn


def get_model_output_shape(model, expected_input_shape):
    return model(torch.rand(expected_input_shape)).data.shape


class ProtoPNet(nn.Module):
    def __init__(self, base_model="vgg16", output_channels=128, prototypes_per_class=10, num_output_classes=200):
        """
        :param base_model: one of 'vgg16', 'vgg19'
        :param output_channels: one of 128, 256, 512
        """
        super().__init__()

        self.pretrained_conv_net = torch.hub.load("pytorch/vision:v0.10.0", base_model, pretrained=True)
        del self.pretrained_conv_net.classifier  # Remove classification layers

        # TODO: support other base models, conditionally construct `expected input_shape`
        conv_output_channels = get_model_output_shape(self.pretrained_conv_net, (1, 3, 224, 224))[1]

        self.additional_layers = nn.Sequential(
            nn.Conv2d(conv_output_channels, output_channels, 1),  # First 1x1 convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1),  # Second 1x1 convolution
            nn.Sigmoid(),
        )

        self.num_prototypes = prototypes_per_class * num_output_classes
        prototype_shape = (self.num_prototypes, output_channels, 1, 1)

        # Each class has a onehot prototype representation
        self.prototype_onehot_class_representation = torch.zeros((num_output_classes, self.num_prototypes))

        # TODO: Xavier initialisation
        self.prototypes = nn.Parameter(torch.randn(prototype_shape))
        self.ones = nn.Parameter(torch.ones(prototype_shape), requires_grad=False)
        self.fully_connected = nn.Linear(self.num_prototypes, num_output_classes, bias=False)

    def forward(self, x):
        x = self.pretrained_conv_net(x)
        x = self.additional_layers(x)

        # Do the prototype stuff! This is basically copied from the original repo, I don't really understand it :(
        distances = self.prototype_distances(x)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        x = self.prototype_activations(distances)

        x = self.fully_connected(x)
        return x, min_distances

    def prototype_distances(self, x):
        x_convolved_prototypes = torch.sum(F.conv2d(x, weight=self.prototypes))
        prototypes_squared = torch.sum(self.prototypes**2, dim=(1, 2, 3))
        intermediate = -2 * x_convolved_prototypes + prototypes_squared.view(-1, 1, 1)

        x2_convolved_patches = F.conv2d(x**2, weight=self.ones)

        distances = F.relu(x2_convolved_patches + intermediate)
        return distances

    def prototype_activations(self, distances, epsilon=0.001):
        distances = distances.view(-1, self.prototypes.shape[0])
        return torch.log((distances + 1) / (distances + epsilon))
