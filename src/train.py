import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AdvancedMNISTCNN

def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations & augmentations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Determine root directory dynamically (points to mnist-production-pipeline/)
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"

    # Load MNIST Dataset
    train_dataset = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize model, loss, and optimizer
    model = AdvancedMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (3 Epochs)
    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_dataset)
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Set up models directory at the project root
    MODELS_DIR = ROOT_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    # Save PyTorch state dict
    torch.save(model.state_dict(), MODELS_DIR / 'mnist_cnn.pth')

    # Ensure model is in eval mode for clean ONNX export
    model.eval()

    # Export to ONNX runtime format
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    onnx_path = MODELS_DIR / 'mnist_cnn.onnx'
    torch.onnx.export(
        model, 
        dummy_input, 
        str(onnx_path),
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully trained, evaluated, and exported to {onnx_path}!")

if __name__ == "__main__":
    train_and_export()