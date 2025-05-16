from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import io

app = FastAPI()

# Allow frontend access (adjust domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic health check
@app.get("/")
async def root():
    return {"message": "Smart Image Classifier API is running."}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load general ImageNet model
general_model = models.resnet50(pretrained=True)
general_model.eval()
with open("imagenet_classes.txt") as f:
    general_labels = [line.strip() for line in f.readlines()]

# Load custom model
checkpoint = torch.load("custom_model.pth", map_location=torch.device("cpu"))
custom_model = models.resnet18(pretrained=False)
custom_model.fc = torch.nn.Linear(custom_model.fc.in_features, len(checkpoint.get('class_names', [])))
custom_model.load_state_dict(checkpoint.get('model_state_dict'))
custom_model.eval()
custom_labels = checkpoint.get('class_names', [])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Predict with general model
        with torch.no_grad():
            gen_output = general_model(input_tensor)
            gen_probs = F.softmax(gen_output, dim=1)
            gen_conf, gen_idx = gen_probs[0].max(0)
            general_result = {
                "label": general_labels[gen_idx.item()],
                "confidence": round(gen_conf.item(), 3)
            }

        # Predict with custom model
        with torch.no_grad():
            cust_output = custom_model(input_tensor)
            cust_probs = F.softmax(cust_output, dim=1)
            cust_conf, cust_idx = cust_probs[0].max(0)
            custom_result = {
                "label": custom_labels[cust_idx.item()],
                "confidence": round(cust_conf.item(), 3)
            }

        # Best prediction
        best = {
            "source": "general" if gen_conf > cust_conf else "custom",
            **(general_result if gen_conf > cust_conf else custom_result)
        }

        return JSONResponse({
            "best_prediction": best,
            "general_model": general_result,
            "custom_model": custom_result
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )