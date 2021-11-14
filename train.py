import os
import time
import torch
import torchvision
from torchvision import transforms
import torch.utils.data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context#不验证ssl证书
from vgg import vgg19_bn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

MODEL_PATH = './models'
MODEL_NAME = 'vgg19_bn'+NUM_EPOCHS+'.pth'#根据epoch的不同最后的模型数据也将不同

# Create model
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

transform = transforms.Compose([
    transforms.RandomCrop(36, padding=4),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),#依照概率对图片进行翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#逐层对图像进行标准化，加快收敛
])


# Load data
dataset = torchvision.datasets.CIFAR10(root='./data',
                                        download=True,
                                        train=True,
                                        transform=transform)

dataset_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)


def main():
    print(f"Train numbers:{len(dataset)}")

    model = vgg19_bn().to(device)

    cast = torch.nn.CrossEntropyLoss().to(device)
    #optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-8)
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum = 0.9)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()

        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * len(dataset)}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1

        # cal train one epoch time
        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + '/' + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + '/' + MODEL_NAME}.")


if __name__ == '__main__':
    main()
