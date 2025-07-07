import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image
import io
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

device = torch.device("cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_mnist.pth", map_location=device))
model.eval()

app = FastAPI()

class ImageInput(BaseModel):
    image_base64: str

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.post("/predict-image")
def predict_image(data: ImageInput):
    try:
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()

        return {"prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
