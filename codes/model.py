import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_V2_M_Weights

class CustomEfficientNetV2(nn.Module):
    def __init__(self, base_model_name='efficientnet_v2_m', num_classes=1, pretrained=True, n_metadata=77):
        super(CustomEfficientNetV2, self).__init__()
        if base_model_name == 'efficientnet_v2_m':
            if pretrained:
                self.base_model = models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            else:
                self.base_model = models.efficientnet_v2_m(weights=None)
                
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        
        self.feature_reduction = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SiLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + n_metadata, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x, metadatas):
        features = self.base_model(x)
        features = self.feature_reduction(features)
        x = torch.cat((features, metadatas), dim=1)
        x = self.classifier(x)
        return x


# Below is for timm lib but timm doesn't have v2 pretrained params

'''
class CustomEfficientNetV2(nn.Module):
    def __init__(self, base_model_name='efficientnetv2_m', num_classes=1, pretrained=True, n_metadata=77, checkpoint_path=None):
        super(CustomEfficientNetV2, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.base_model.reset_classifier(0)
        num_features = self.base_model.num_features
        
        self.feature_reduction = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SiLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + n_metadata, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x, metadatas):
        features = self.base_model(x)
        features = self.feature_reduction(features)
        x = torch.cat((features, metadatas), dim=1)
        x = self.classifier(x)
        return x

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
'''
