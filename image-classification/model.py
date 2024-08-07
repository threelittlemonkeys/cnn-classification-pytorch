from utils import *

class cnn(nn.Module):

    def __init__(self, num_labels):

        super().__init__()

        # architecture
        self.conv1 = nn.Conv2d(3, 27, 5, padding = "same")
        self.conv2 = nn.Conv2d(27, 81, 5, padding = "same")
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear((IMG_WIDTH // 2 // 2) * (IMG_HEIGHT // 2 // 2) * 81, num_labels)
        self.softmax = nn.LogSoftmax(1)

        if CUDA:
            self = self.cuda()

    def forward(self, x):

        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = torch.flatten(h, 1)
        h = self.fc(h)
        y = self.softmax(h)

        return y
