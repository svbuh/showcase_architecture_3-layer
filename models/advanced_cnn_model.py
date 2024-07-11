import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        # https://de.mathworks.com/help/deeplearning/ref/resnet18.html
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 512)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path='advanced_cnn.pth'):
    model = AdvancedCNN()
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
