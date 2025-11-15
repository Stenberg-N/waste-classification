import torch, time
import torch.optim as optim
from src.model import get_model
from src.dataloader import split_dataset, get_dataloaders
from torch.utils.tensorboard import SummaryWriter
from configs.paths import LOGS_DIR, MODELS_DIR

if __name__ == '__main__':
    split_dataset()

    stage1_epochs = 15
    stage2_epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name='mobilenetv4_hybrid_medium.e500_r224_in1k', num_classes=6, pretrained=True).to(device)

    train_loader, validation_loader, _ = get_dataloaders()
    writer = SummaryWriter(LOGS_DIR)

    criterion = torch.nn.CrossEntropyLoss()
    best_validation_loss = float('inf')

    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.0001)

    for epoch in range(stage1_epochs):
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            start = time.time()
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_validation_loss = validation_loss / len(validation_loader)
        average_train_loss = train_loss / len(train_loader)
        validation_accuracy = 100 * correct / total

        writer.add_scalar('Loss/train_stage1', average_train_loss, epoch)
        writer.add_scalar('Loss/validation_stage1', average_validation_loss, epoch)
        writer.add_scalar('Accuracy/validation_stage1', validation_accuracy, epoch)

        print(f'[STAGE 1] Epoch {epoch+1}/{stage1_epochs}: Train Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), (MODELS_DIR / f'stage1_best.pth'))

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(stage2_epochs):
        model.train()
        train_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        average_validation_loss = validation_loss / len(validation_loader)
        average_train_loss = train_loss / len(train_loader)
        validation_accuracy = 100 * correct / total
        scheduler.step(average_validation_loss)

        writer.add_scalar('Loss/train_stage2', average_train_loss, epoch)
        writer.add_scalar('Loss/validation_stage2', average_validation_loss, epoch)
        writer.add_scalar('Accuracy/validation_stage2', validation_accuracy, epoch)

        print(f'[STAGE 2] Epoch {epoch+1}/{stage2_epochs}: Train Loss: {average_train_loss:.4f}, Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), (MODELS_DIR / f'stage2_best.pth'))

    writer.close()