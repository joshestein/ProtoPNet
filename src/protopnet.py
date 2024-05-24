import torch
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

        num_prototypes = prototypes_per_class * num_output_classes
        # Each class has a onehot prototype representation
        self.prototype_onehot_class_representation = torch.zeros((num_output_classes, num_prototypes))

        # TODO: Xavier initialisation
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, output_channels, 1, 1))
        self.fully_connected = nn.Linear(num_prototypes, num_output_classes, bias=False)

    def forward(self, x):
        x = self.pretrained_conv_net(x)
        x = self.additional_layers(x)
        return x
