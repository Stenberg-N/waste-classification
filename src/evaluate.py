import torch
from src.model import get_model
from PIL import Image
from torchvision import transforms
from src.dataloader import get_dataloaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from configs.paths import MODELS_DIR

def evaluate_test_set(checkpoint_path=(MODELS_DIR / 'stage2_best.pth')):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name='densenet201', num_classes=6, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'])
    display.plot()
    plt.show()

def predict_image(image_path, checkpoint_path=(MODELS_DIR / 'stage2_best.pth'), image_size=224):
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name='densenet201', num_classes=6)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

    predicted_class = classes[predicted.item()]
    print(f'Predicted: {predicted_class} (Confidence: {confidence * 100:.2f}%)')
    return predicted_class, confidence

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(image_path)
    else:
        evaluate_test_set()
        print('Usage: python src/evaluate.py <image_path>')