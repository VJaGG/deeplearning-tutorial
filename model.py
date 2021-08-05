'''
@File    : model.py
@Modify Time     @Author    @Version    @Desciption
------------     -------    --------    -----------
2021/8/5 15:13   WuZhiqiang     1.0        None 
'''
import timm
from common import *


class EfficientB4(nn.Module):
    def __init__(self, arch='tf_efficientnet_b4', dim=512, num_classes=101, pretrained=True):
        super(EfficientB4, self).__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.linear = nn.Linear(final_in_features, dim)
        self.silu = nn.SiLU()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        x = self.silu(x)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    model = EfficientB4()
    from torchsummary import summary
    summary(model, input_size=(3, 512, 512), device='cpu')