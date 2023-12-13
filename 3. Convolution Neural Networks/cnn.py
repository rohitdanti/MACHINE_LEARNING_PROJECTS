# import torch and other necessary modules from torch
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, precision_score

# import torchvision and other necessary modules from torchvision
import torchvision
from torchvision import transforms
from torchvision import datasets


# recommended preprocessing steps: resize to square -> convert to tensor -> normalize the image
# if you are resizing, 100 is a good choice otherwise GradeScope will time out
# you could use Compose (https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) from transforms module to handle preprocessing more conveniently
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize images to 100x100 (change the size to your preferred value)
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image values (adjust mean and std if needed)
])


'''from google.colab import drive
drive.mount('/content/gdrive')
dataset = datasets.ImageFolder(root = "/content/gdrive/MyDrive/petimages", transform=transform)
'''
dataset = datasets.ImageFolder(root="./petimages", transform=transform)


# now we need to split the data into training set and evaluation set
# use 20% of the dataset as test
test_size = int(0.2 * len(dataset))  # using 20% of the pet images dataset for testing
train_size = len(dataset) - test_size  # using the other 80% of the pet images dataset for training
test_set, train_set = random_split(dataset, [test_size, train_size])

# model hyperparameter
learning_rate = 0.000002
batch_size = 32
epoch_size = 20

# test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
# train_set = torch.utils.data.Subset(dataset, range(n_test, len(dataset)))  # take the rest
n_test = int(0.1 * len(dataset)) #setting the test set size to be 10%
test_set = Subset(dataset, range(n_test)) #creating a subset for test set, here the first 10%
train_set = Subset(dataset, range(n_test, len(dataset))) # creating a subset for the train set, here the rest 90%
# Creating DataLoaders for training and testing
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #to improve he training, shuffle is set to true. so data is shuffled during each epoch
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False) #order same during evaluation so shuffle is set to false

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        x = self.fc3(x)
        return x

'''
# Check if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))  # Use the first GPU if there are multiple GPUs
else:
    device = torch.device("cpu")
    print("Using CPU")
cnn = CNN().to(device)'''

device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU



criterion = nn.CrossEntropyLoss()
# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = optim.Adam(cnn.parameters(), lr=0.0003)

# start model training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size): # begin with trying 10 epochs

    running_loss = 0.0 # you can print out average loss per batch every certain batches

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, labels = data
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()
        # forward -> compute loss -> backward propogation -> optimize (see tutorial mentioned in main documentation)
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print some statistics
        running_loss += loss.item()  # add loss for current batch
        if i % 100 == 99:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss  = 0.0

print('Finished Training')

# Your code for model training (provided in the previous section) goes here

# Your model has already been trained at this point

# Evaluation on the evaluation set
ground_truth = []  # Lists to store ground truth
prediction = []  # Lists to store predicted labels

cnn.eval()  # Turn on evaluation mode (a good practice)

with torch.no_grad():  # Since we're not training, we don't need to calculate gradients for our outputs, so turn on no_grad mode
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels.tolist()  # Convert labels to a list and append to ground_truth
        # Calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        # The class with the highest logit is what we choose as the prediction
        _, predicted = torch.max(outputs, 1)
        prediction += predicted.tolist()  # Convert predicted labels to a list and append to prediction

# Use scikit-learn to calculate the scores
accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='weighted',zero_division="warn")
precision = precision_score(ground_truth, prediction, average='weighted')

# Print or use these metrics as needed
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')

