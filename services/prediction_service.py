import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from models.simple_cnn_model import load_model


class PredictionService:
    def __init__(self):
        self.model = load_model()
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image).unsqueeze(0)
        return image

    def predict_from_url(self, url):
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = self.preprocess_image(image)
        outputs = self.model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = self.classes[predicted.item()]
        return class_name
