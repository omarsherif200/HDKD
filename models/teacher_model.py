from blocks import *
from timm.models.registry import register_model
class TeacherModel(nn.Module):
    """
        Args:
        image_size (int): Size of input images. Default: 224
        in_channel (int): Number of channels in the input images. Default: 3
        num_blocks (List(int)): Number of blocks at each stage. Default: [2, 2, 3, 3]
        channels (List(int)): The output dimension at each stage. Default: [64, 96, 192, 256]
        num_classes (int): Number of classes in the dataset. Default: 7 (for HAM-10000 dataset).

    """
    def __init__(self, image_size, in_channel, num_blocks, channels, num_classes=7, **kwargs):
        super().__init__()
        ih,iw= image_size
        block={'stem':conv_3x3_bn,'MBConv':MBConv,'MBCSA':MBCSA}
        self.stage1 = self._make_block(block['stem'],in_channel,channels[0],3,num_blocks[0])
        self.stage2 = self._make_block(block['MBCSA'],channels[0],channels[1],3,num_blocks[1])
        self.stage3 = self._make_block(block['MBCSA'],channels[1],channels[2],3,num_blocks[2])
        self.pool= nn.AdaptiveAvgPool2d(1)
        self.flatten_layer=nn.Flatten()
        self.fn1= nn.Linear(channels[2],128)
        self.classifier= nn.Linear(128,num_classes)


    def forward(self,x):
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.pool(x)
        x=self.flatten_layer(x)
        x=self.fn1(x)
        out=self.classifier(x)
        return out

    def _make_block(self,block,inp,oup,kernel_size,depth):
        layers=nn.ModuleList([])
        for i in range(depth):
            if i==0:
                layers.append(block(inp,oup,kernel_size,downsample=True))
            else:
                layers.append(block(oup,oup,kernel_size))

        return nn.Sequential(*layers)

@register_model
def teacher_model(**kwargs):
    num_blocks = [6, 6, 9]
    channels = [64,96,192]
    return TeacherModel((224, 224), 3, num_blocks, channels, **kwargs)
