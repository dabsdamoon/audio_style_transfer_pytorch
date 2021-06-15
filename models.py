import torch.nn as nn

##### Define model class
class SoundStyleTransfer(nn.Module):
    def __init__(self, in_channels, out_channels, filter_shape = (1,11)):
        
        super(SoundStyleTransfer, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.conv_layer = nn.Conv2d(in_channels, out_channels, filter_shape)
        self.activation = nn.ReLU()
    
    def forward(self, input_tensor):
        output_tensor = self.conv_layer(input_tensor)
        output_tensor = self.activation(output_tensor)
        return output_tensor