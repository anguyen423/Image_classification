from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import models, transforms
import io
import torch.nn.functional as F

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Common transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load general model (ImageNet)
general_model = models.resnet50(pretrained=True)
general_model.eval()
with open("imagenet_classes.txt") as f:
    general_labels = [line.strip() for line in f.readlines()]

# Load custom model
checkpoint = torch.load("custom_model.pth", map_location=torch.device("cpu"))
custom_model = models.resnet18(pretrained=False)
custom_model.fc = torch.nn.Linear(custom_model.fc.in_features, len(checkpoint['class_names']))
custom_model.load_state_dict(checkpoint['model_state_dict'])
custom_model.eval()
custom_labels = checkpoint['class_names']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # General model prediction
        with torch.no_grad():
            general_output = general_model(input_tensor)
            general_probs = F.softmax(general_output, dim=1)
            gen_conf, gen_idx = general_probs[0].max(0)
            general_result = {
                "label": general_labels[gen_idx.item()],
                "confidence": round(gen_conf.item(), 3)
            }

        # Custom model prediction
        with torch.no_grad():
            custom_output = custom_model(input_tensor)
            custom_probs = F.softmax(custom_output, dim=1)
            cust_conf, cust_idx = custom_probs[0].max(0)
            custom_result = {
                "label": custom_labels[cust_idx.item()],
                "confidence": round(cust_conf.item(), 3)
            }

        # Optional: Pick best (confidence-based)
        if gen_conf > cust_conf:
            best = {"source": "general", **general_result}
        else:
            best = {"source": "custom", **custom_result}

        return JSONResponse({
            "best_prediction": best,
            "general_model": general_result,
            "custom_model": custom_result
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})