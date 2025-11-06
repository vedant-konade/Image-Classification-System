import torch
from torchvision import transforms
from PIL import Image
from models.cnn import SimpleCifarCNN

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_model(model_path="outputs/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCifarCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_image(img_path, model, device):
    tfm = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                            std=[0.2470,0.2435,0.2616]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0)
        cls_id = int(prob.argmax().cpu())
        return CLASSES[cls_id], float(prob[cls_id].cpu())
if __name__ == "__main__":
    model, device = load_model("outputs/best_model.pt")
    test_image = "test3.jpeg"  # <-- replace with your image path

    label, prob = predict_image(test_image, model, device)
    print(f"Predicted: {label} ({prob:.4f})")
