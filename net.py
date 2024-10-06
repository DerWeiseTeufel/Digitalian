

from torch import nn
# Igore warnings


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    in_c = 16
    self.conv_layer1 = nn.Conv2d(kernel_size=3, padding=1, in_channels = 1, out_channels = 16) # ((28-3) + 2)/1 + 1 = 28
    self.norm1 = nn.BatchNorm2d(16)
    self.mxpool1 = nn.MaxPool2d(3)# OUT = 9
    self.conv_layer2 = nn.Conv2d(kernel_size=5, padding=1, in_channels = 16, out_channels=128)
    self.norm2 = nn.BatchNorm2d(128)
    #max pool -> 3
    self.conv_layer3 = nn.Conv2d(kernel_size=3, padding=1, in_channels = 128, out_channels=512)
    self.norm3 = nn.BatchNorm2d(512)
    self.aveg = nn.AvgPool2d(2)

    # avgpool -> 1

    self.activ = nn.LeakyReLU()
    self.lin1 = nn.Linear(in_features = 512, out_features = 1024)
    self.norm8 = nn.BatchNorm1d(1024)

    self.lin2 = nn.Linear(1024, 1024)
    self.norm9 = nn.BatchNorm1d(1024)

    self.lin3 = nn.Linear(1024, 1024)
    self.norm10 = nn.BatchNorm1d(1024)

    self.lin4 = nn.Linear(1024, 10)

  def forward(self, x):
    x = self.activ(self.mxpool1(self.norm1(self.conv_layer1(x))))
    x = self.activ(self.mxpool1(self.norm2(self.conv_layer2(x))))



    x = self.activ(self.aveg(self.norm3(self.conv_layer3(x))))

    x = x.squeeze()
    x = self.activ(self.norm8(self.lin1(x)))
    x = self.activ(self.norm9(self.lin2(x)))
    x = self.activ(self.norm10(self.lin3(x)))
    x = self.lin4(x)
    return x



class Net2(nn.Module):
  def __init__(self):
    super(Net2, self).__init__()
    in_c = 16
    self.conv_layer1 = nn.Conv2d(kernel_size=3, padding=1, in_channels = 1, out_channels = 16) # ((28-3) + 2)/1 + 1 = 28
    self.mxpool1 = nn.MaxPool2d(3)# OUT = 9
    self.conv_layer2 = nn.Conv2d(kernel_size=5, padding=1, in_channels = 16, out_channels=128)
    #max pool -> 3
    self.conv_layer3 = nn.Conv2d(kernel_size=3, padding=1, in_channels = 128, out_channels=512)
    self.aveg = nn.AvgPool2d(2)

    # avgpool -> 1

    self.activ = nn.LeakyReLU()
    self.lin1 = nn.Linear(in_features = 512, out_features = 1024)

    self.lin2 = nn.Linear(1024, 1024)

    self.lin3 = nn.Linear(1024, 1024)

    self.lin4 = nn.Linear(1024, 10)

  def forward(self, x):
    x = self.activ(self.mxpool1(self.conv_layer1(x)))
    x = self.activ(self.mxpool1(self.conv_layer2(x)))



    x = self.activ(self.aveg((self.conv_layer3(x))))

    x = x.squeeze()
    x = self.activ((self.lin1(x)))
    x = self.activ((self.lin2(x)))
    x = self.activ((self.lin3(x)))
    x = self.lin4(x)
    return x



