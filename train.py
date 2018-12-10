import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from DeXpression.Model import Block1, Block2, Block3

# defining hyper-parameters
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_DATA_PATH = "./train_set"

# creating training and test tensors
transform = transforms.Compose(
    [transforms.CenterCrop(480),
     transforms.Resize(224),
     transforms.Grayscale(3),
     transforms.RandomRotation(2),
     transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
classes = ('Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise',)

block1 = Block1()
block2 = Block2()
block3 = Block3()
save_block1 = block1
save_block2 = block2
save_block3 = block3
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(block3.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer2 = optim.SGD(block2.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer3 = optim.SGD(block1.parameters(), lr=0.001, momentum=0.9, nesterov=True)
for epoch in range(EPOCHS):
    running_loss = 0.0
    running_loss1 = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        outputs = block1(inputs)
        outputs = block2(outputs, outputs)
        outputs = block3(outputs, outputs)
        # print(outputs)
        # print(labels)
        # print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(8)))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        # printing statistics
        running_loss1 += loss.item()
        # running_loss += loss.item()
        # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        # running_loss = 0.0
        # printing every 10 mini-batches
        if i % 10 == 9:
            print('EPOCH: %d, BATCH NUMBER: %5d LOSS: %.5f' % (epoch + 1, i + 1, running_loss1 / 10))
            running_loss1 = 0.0
            # saving every mini-batch
            torch.save({'BLOCK_1_state_dict': save_block1.state_dict(),
                        'BLOCK_2_state_dict': save_block2.state_dict(),
                        'BLOCK_3_state_dict': save_block3.state_dict()
                        }, 'last_model_state.pth')
print("Finished Training")
# saving the final model
torch.save({'BLOCK_1_state_dict': save_block1.state_dict(),
            'BLOCK_2_state_dict': save_block2.state_dict(),
            'BLOCK_3_state_dict': save_block3.state_dict()
            }, 'last_model_state.pth')
