import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)


class AbstractNet(nn.Module):

    def __init__(self):
        super(AbstractNet, self).__init__()
        self.number_of_parameters = None

    def update(self):
        self.number_of_parameters = 0
        for elt in self.parameters():
            self.number_of_parameters += len(elt.view(-1))
        if Settings().tensor_scalar_type == torch.cuda.FloatTensor:
            logger.info("Setting neural network type to CUDA.")
            self.cuda()
        elif Settings().tensor_scalar_type == torch.DoubleTensor:
            logger.info("Setting neural network type to Double.")
            self.double()
        net_name = str(type(self))
        net_name = net_name[net_name.find('_nets.')+6:net_name.find('>')-1]
        logger.info("The nn {} has".format(net_name), self.number_of_parameters, "weights.")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_gradient(self):
        out = np.zeros(self.number_of_parameters)
        pos = 0
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    out[pos:pos+len(layer.weight.view(-1))] = layer.weight.grad.view(-1).cpu().data.numpy()
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    out[pos:pos+len(layer.bias.view(-1))] = layer.bias.grad.view(-1).cpu().data.numpy()
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        return out

    def set_parameters(self, nn_parameters):
        """
        sets parameters from the given (flat) variable (should use state_dict)
        """
        pos = 0
        # logger.info("Setting net param", nn_parameters.cpu().data.numpy()[0])
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    layer.weight.data = nn_parameters[pos:pos+len(layer.weight.view(-1))].view(layer.weight.size()).data
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    layer.bias.data = nn_parameters[pos:pos+len(layer.bias.view(-1))].view(layer.bias.size()).data
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        self.assert_rank_condition()

    def get_parameters(self):
        """"
        returns a numpy array with the flattened weights
        """
        out = np.zeros(self.number_of_parameters)
        pos = 0
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    # logger.info(layer, layer.weight.cpu().data.numpy().shape)
                    out[pos:pos+len(layer.weight.view(-1))] = layer.weight.view(-1).cpu().data.numpy()
                    pos += len(layer.weight.view(-1))
            except AttributeError:
                pass
            try:
                if layer.bias is not None:
                    out[pos:pos+len(layer.bias.view(-1))] = layer.bias.view(-1).cpu().data.numpy()
                    pos += len(layer.bias.view(-1))
            except AttributeError:
                pass
        return out

    def assert_rank_condition(self):
        """
        Fletcher condition on generative networks,
        so that the image is (locally) a submanifold of the space of observations
        """
        #return
        for layer in self.layers:
            try:
                if layer.weight is not None:
                    np_weight = layer.weight.data.numpy()
                    if len(np_weight.shape) == 4: # for convolution layers
                        a, b, c, d = np_weight.shape
                        np_weight = np_weight.reshape(a, b * c * d)
                    a, b = np_weight.shape
                    rank = np.linalg.matrix_rank(layer.weight.data.numpy())
                    assert rank == min(a, b), "Weight of layer does not have full rank {}".format(layer)
            except AttributeError:
                pass


class ScalarNet(AbstractNet):

    def __init__(self, in_dimension=2, out_dimension=4):
        super(ScalarNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dimension, in_dimension),
            nn.Tanh(),
            nn.Linear(in_dimension, 10*out_dimension, bias=True),
            nn.Tanh(),
            nn.Linear(10*out_dimension, 5*out_dimension, bias=True),
            nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension),
            # nn.Tanh(),
            # nn.Linear(out_dimension, out_dimension),
            # nn.Tanh(),
            nn.Linear(5*out_dimension, out_dimension),
            nn.Tanh()
            ])
        self.update()


# This net automatically outputs 64 x 64 images. (to be improved)
class ImageNet2d(AbstractNet):

    def __init__(self, in_dimension=2):
        super(ImageNet2d, self).__init__()
        ngf = 2
        self.layers = nn.ModuleList([
            nn.Linear(in_dimension, in_dimension),
            nn.Tanh(), # was added 29/03 12h
            nn.ConvTranspose2d(in_dimension, 16 * ngf, 4, stride=4),
            # nn.ELU(), # was removed, same day 14h
            # nn.ConvTranspose2d(32 * ngf, 16 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=16*ngf),
            nn.ConvTranspose2d(16 * ngf, 8 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=8*ngf),
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=4*ngf),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=2*ngf),
            nn.ConvTranspose2d(2 * ngf, 1, 2, stride=2),
            nn.ELU()
        ])
        self.update()

    def forward(self, x):
        a = x.size()
        # if len(a) == 2:
        #     x = x.unsqueeze(2).unsqueeze(3)
        # else:
        #     x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                if len(a) == 2:
                    x = x.unsqueeze(2).unsqueeze(3)
                else:
                    x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                x = layer(x)
        if len(a) == 2:
            return x.squeeze(1)
        else:
            return x.squeeze(1).squeeze(0)

# This net automatically outputs 128 x 128 images. (to be improved)
class ImageNet2d128(AbstractNet):

    def __init__(self, in_dimension=2):
        super(ImageNet2d128, self).__init__()
        ngf = 2
        self.layers = nn.ModuleList([
            nn.Linear(in_dimension, in_dimension),
            nn.Tanh(), # was added 29/03 12h
            nn.ConvTranspose2d(in_dimension, 32 * ngf, 4, stride=4),
            # nn.ELU(), # was removed, same day 14h
            # nn.ConvTranspose2d(32 * ngf, 16 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=16*ngf),
            nn.ConvTranspose2d(32 * ngf, 16 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=8*ngf),
            nn.ConvTranspose2d(16 * ngf, 8 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=4*ngf),
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=2*ngf),
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=2*ngf),
            nn.ConvTranspose2d(2 * ngf, 1, 2, stride=2),
            nn.ELU()
        ])
        self.update()

    def forward(self, x):
        a = x.size()
        # if len(a) == 2:
        #     x = x.unsqueeze(2).unsqueeze(3)
        # else:
        #     x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x)
                if len(a) == 2:
                    x = x.unsqueeze(2).unsqueeze(3)
                else:
                    x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                x = layer(x)
        if len(a) == 2:
            return x.squeeze(1)
        else:
            return x.squeeze(1).squeeze(0)

# This net automatically outputs 64 x 64 x 64 images. (to be improved)
class ImageNet3d(AbstractNet):

    def __init__(self, in_dimension=2):
        super(ImageNet3d, self).__init__()
        ngf = 2
        self.layers = nn.ModuleList([
            nn.ConvTranspose3d(in_dimension, 32 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            nn.ConvTranspose3d(32 * ngf, 16 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            nn.ConvTranspose3d(16 * ngf, 8 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            nn.ConvTranspose3d(8 * ngf, 4 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            nn.ConvTranspose3d(4 * ngf, 2 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            nn.ConvTranspose3d(2 * ngf, 1, 2, stride=2, bias=False),
            nn.ELU()
        ])
        self.update()

    def forward(self, x):
        a = x.size()
        if len(a) == 2:
            x = x.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        for layer in self.layers:
            x = layer(x)
        if len(a) == 2:
            return x.squeeze(1)
        else:
            return x.squeeze(1).squeeze(0)


# This net automatically outputs 28 x 28 images. (to be improved)
class MnistNet(AbstractNet):

    def __init__(self, in_dimension=2):
        super(MnistNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dimension, 8*2*2),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            # nn.ELU(), # was removed, same day 14h
            # nn.ConvTranspose2d(32 * ngf, 16 * ngf, 2, stride=2, bias=False),
            nn.ELU(),
            # nn.BatchNorm2d(num_features=16*ngf),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ELU(),
            # nn.BatchNorm2d(num_features=8*ngf),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.ELU(),
        ])
        self.update()

    def forward(self, x):
        a = x.size()
        # if len(a) == 2:
        #     x = x.unsqueeze(2).unsqueeze(3)
        # else:
        #     x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        for i, layer in enumerate(self.layers):
            #logger.info(x.size())
            if i == 0:
                x = layer(x)
                if len(a) == 2:
                    x = x.unsqueeze(2).unsqueeze(3)
                else:
                    x = x.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = x.view(len(x), 8, 2, 2)
            else:
                x = layer(x)
        if len(a) == 2:
            return x.squeeze(1)
        else:
            return x.squeeze(1).squeeze(0)


class DiffeoNet(AbstractNet):

    def __init__(self, in_dimension=10, nb_cp=10):
        super(DiffeoNet, self).__init__()
        self.nb_cp = nb_cp
        self.fc1 = nn.Linear(in_dimension, 2 * nb_cp * Settings().dimension)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(2 * nb_cp * Settings().dimension, 2 * nb_cp * Settings().dimension)
        self.elu2 = nn.ELU()
        self.tanh = nn.Tanh()
        self.update()

    def forward(self, x):
        x = self.elu1(self.fc1(x))
        x = self.fc2(x)
        x = x.view(len(x), 2, self.nb_cp, Settings().dimension)
        a = torch.clamp(x[:, 0, :, :], 0, 28)# control points
        b = 10 * self.tanh(x[:, 1, :, :])# momenta
        return torch.stack([a,b], 1)


class DiffeoMnistDConvNet(AbstractNet):

    def __init__(self, in_dimension=10):
        super(DiffeoMnistDConvNet, self).__init__()
        self.deconvolutions = nn.ModuleList([
            nn.ConvTranspose2d(in_dimension, 16, stride=2, kernel_size=2),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, stride=2, kernel_size=2),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 2, stride=2, kernel_size=2)
        ])
        self.update()

    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        for layer in self.deconvolutions:
            x = layer(x)
        x = x.view(len(x), 8*8, 2)
        return x


class MnistDiscrimator(AbstractNet):

    def __init__(self, noise_in_dimension=10):
        super(MnistDiscrimator, self).__init__()
        self.convolution = nn.ModuleList([
            nn.Conv2d(1, 8, kernel_size=3, stride=3),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=3),
            nn.ELU()
        ])
        self.fc1 = nn.Linear(144, 20)
        self.fc2 = nn.Linear(20, 1)
        self.update()

    def forward(self, x):
        for layer in self.convolution:
            x = layer(x)
        x = x.view(len(x), -1)
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class MnistEncoder(AbstractNet):

    def __init__(self, out_dimension=10):
        super(MnistEncoder, self).__init__()
        self.convolution = nn.ModuleList([
            nn.Conv2d(1, 4, kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(4, 8, kernel_size=2, stride=2),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2),
            nn.ELU()
        ])
        self.fc1 = nn.Linear(144, out_dimension)
        self.fc2 = nn.Linear(out_dimension, out_dimension)
        self.fc3 = nn.Linear(out_dimension, out_dimension)
        self.update()

    def forward(self, x):
        for layer in self.convolution:
            x = layer(x)
        x = x.view(len(x), -1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x