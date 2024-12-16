import torch
from torchvision import transforms
from PIL import Image
import os

# Determine the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = '64_5e-05_10_0.0_reduce-on-plateau'
model = torch.load(os.path.join(base_dir, 'models', 'full_models', f'{model_name}_full.pth'))
model.to(device)  # Move the model to the correct device
model.eval()


# Prepare the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define your class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Directory containing test images
test_dir = "../data/test_images"

# Loop through all images in the test directory
for image_name in os.listdir(test_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        image_path = os.path.join(test_dir, image_name)

        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            input_tensor = input_tensor.to(device)  # Move input to the correct device
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

        print(f"Image: {image_name}")
        print(f"Predicted class: {class_names[predicted.item()]}")
        print()
