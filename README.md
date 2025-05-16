# Smart Image Classifier

This app uses both a pretrained general model (ImageNet) and a custom-trained model to recognize images uploaded by users.

## How to Use
1. Upload a photo in the frontend
2. The app returns predictions from:
   - General model (e.g., ResNet50)
   - Custom model (your own categories)
3. Displays both results and the best match.

## Deployment
Uses Render + FastAPI + HTML frontend.