import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG16_Weights


def get_model_output_shape(model, expected_input_shape):
    return model(torch.rand(expected_input_shape)).data.shape


class ProtoPNet(nn.Module):
    def __init__(self, base_model="vgg16", output_channels=128, prototypes_per_class=10, num_output_classes=200):
        """
        :param base_model: one of 'vgg16', 'vgg19'
        :param output_channels: one of 128, 256, 512
        """
        super().__init__()

        self.pretrained_conv_net = torch.hub.load(
            "pytorch/vision:v0.10.0", base_model, weights=VGG16_Weights.IMAGENET1K_V1
        )
        # Remove VGG classification layers
        self.pretrained_conv_net = self.pretrained_conv_net.features

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
        self.prototype_onehot_class_representation = torch.zeros((self.num_prototypes, num_output_classes))
        for i in range(self.num_prototypes):
            self.prototype_onehot_class_representation[i, i // prototypes_per_class] = 1

        self.prototypes = nn.Parameter(torch.randn(prototype_shape))
        self.ones = nn.Parameter(torch.ones(prototype_shape), requires_grad=False)
        self.fully_connected = nn.Linear(self.num_prototypes, num_output_classes, bias=False)

        self._initialise_weights()

    def forward(self, x):
        x = self.pretrained_conv_net(x)
        x = self.additional_layers(x)

        # Do the prototype stuff! This is basically copied from the original repo, I don't really understand it :(
        distances = self.prototype_distances(x)
        min_distances = -F.max_pool2d(-distances, kernel_size=(distances.size()[2], distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        x = self.prototype_activations(min_distances)
        x = self.fully_connected(x)

        return x, min_distances

    def prototype_distances(self, x):
        x_convolved_prototypes = torch.sum(F.conv2d(x, weight=self.prototypes))
        prototypes_squared = torch.sum(self.prototypes**2, dim=(1, 2, 3))
        intermediate = -2 * x_convolved_prototypes + prototypes_squared.view(-1, 1, 1)

        x2_convolved_patches = F.conv2d(x**2, weight=self.ones)

        distances = F.relu(x2_convolved_patches + intermediate)
        return distances

    @staticmethod
    def prototype_activations(distances, epsilon=0.001):
        return torch.log((distances + 1) / (distances + epsilon))

    def warm(self):
        """Warm up. The pretrained network is frozen, and only train _all_ the additional prototype layers."""
        for param in self.pretrained_conv_net.parameters():
            param.requires_grad = False

        for param in self.additional_layers.parameters():
            param.requires_grad = True

        self.prototypes.requires_grad = True

        for param in self.fully_connected.parameters():
            param.requires_grad = True

    def all_layers_joint_learning(self):
        """Learning all the layers together, including the pre-trained network layers."""
        for param in self.pretrained_conv_net.parameters():
            param.requires_grad = True

        for param in self.additional_layers.parameters():
            param.requires_grad = True

        self.prototypes.requires_grad = True

        for param in self.fully_connected.parameters():
            param.requires_grad = True

    def convex_optimisation_last_layer(self):
        """Only optimising the last layer."""
        for param in self.pretrained_conv_net.parameters():
            param.requires_grad = False

        for param in self.additional_layers.parameters():
            param.requires_grad = False

        self.prototypes.requires_grad = False

        for param in self.fully_connected.parameters():
            param.requires_grad = True

    def _initialise_weights(self):
        """Kaiming initialisation of additional convolutional layers. The fully connected layer weights are
        initialised to 1 if its output logit is connected to a prototype of the same class, otherwise -0.5."""
        for layer in self.additional_layers.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        positive_prototype_locations = torch.t(self.prototype_onehot_class_representation)
        negative_prototype_locations = 1 - positive_prototype_locations
        self.fully_connected.weight.data.copy_(1 * positive_prototype_locations + -0.5 * negative_prototype_locations)
