import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()   # remove classifier
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(img):
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img)

    feat = feat.cpu().numpy()[0]
    return feat / np.linalg.norm(feat)
