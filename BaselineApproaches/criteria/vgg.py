import torch
import torchvision.models as models
import torchvision.transforms as transforms

class VGG16:

    def __init__(self):
        self.model = models.vgg16(weights='IMAGENET1K_V1')
        if torch.cuda.is_available():
            self.model.cuda()
        # self.estimator.eval()
        # self.pp2 = models.IMAGENET1K_V1.transforms
        self.pp = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                    )
        ])
    
    def pre_process(self, batch):
        batch = self.pp(batch)
        return batch

    def forward(self, batch):
        batch = self.pre_process(batch)
        res = self.model(batch)
        return res