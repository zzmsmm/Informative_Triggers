import torch.nn as nn
import torch.nn.functional as F

from models.ew_layers import EWLinear, EWConv2d

class VGG16(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.conv1 = EWConv2d(3, 64, 3, padding=1)
        self.conv2 = EWConv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = EWConv2d(64, 128, 3, padding=1)
        self.conv4 = EWConv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = EWConv2d(128, 128, 3, padding=1)
        self.conv6 = EWConv2d(128, 128, 3, padding=1)
        self.conv7 = EWConv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = EWConv2d(128, 256, 3, padding=1)
        self.conv9 = EWConv2d(256, 256, 3, padding=1)
        self.conv10 = EWConv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = EWConv2d(256, 512, 3, padding=1)
        self.conv12 = EWConv2d(512, 512, 3, padding=1)
        self.conv13 = EWConv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = EWLinear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = EWLinear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = EWLinear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

    # for exponential weighting
    def enable_ew(self, t):
        for name, param in self.named_parameters():
            if isinstance(name, EWConv2d) or isinstance(name, EWLinear):
                name.enable(t)

def vgg16(**kwargs):
    return VGG16(**kwargs)