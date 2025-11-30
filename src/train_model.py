import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time


def main():
    # 1. הגדרות (Settings)
    # שנו את זה לנתיב המלא במחשב שלכם איפה ששמרתם את התמונות
    # למשל: "C:/Projects/WildlifeMonitor/data"
    data_dir = '../data'

    # בדיקה אם יש כרטיס מסך חזק (GPU) או משתמשים במעבד (CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. הכנת התמונות (Transforms)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # טעינת המידע
    # שימו לב: אם אין לכם תמונות בתיקיות עדיין - זה יזרוק שגיאה
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    # num_workers=0 חשוב מאוד ב-Windows כדי למנוע קריסות!
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Classes found: {class_names}")

    # 3. בניית המודל (ResNet50)
    print("Building model...")
    model = models.resnet50(pretrained=True)

    # הקפאת שכבות (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False

    # החלפת הראש
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 4. אימון
    print("Starting training...")
    since = time.time()

    # נקטין ל-3 אפוקים רק בשביל הבדיקה הראשונית במחשב האישי
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 5. שמירה
    save_path = '../models/species_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # החלק הזה קריטי ב-PyCharm! בלעדיו הקוד לא ירוץ טוב
    main()