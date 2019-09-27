import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import pickle as pkl

data = pkl.load(open('mnist-hw1.pkl', 'rb'))
train, test = shuffle(data['train'], random_state=42), shuffle(data['test'], random_state=42)

train, dev, test = train[:-1024], train[-1024:], test

def rescale_rgb(x):
    x = ((x*64)+63)/256
    return x

def plot_canvas(X, idx=0, dim=28, classes=10):
    canvas = np.zeros((dim*classes, classes*dim, 3))
    
    for i in range(classes):
        for j in range(classes):
            canvas[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = rescale_rgb(X[np.random.randint(len(X))])

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(canvas)
    plt.show()

# Plot a few examples
num_samples_in_train = 1
train = train[:num_samples_in_train]
plot_canvas(train, classes=1)

num_epochs = 50
batch_size = 128
num_batches_train = len(train) // batch_size
num_batches_dev = len(dev) // batch_size
num_batches_test = len(test) // batch_size

get_slice = lambda i, size: range(i * size, (i + 1) * size)


class MaskedConv2d(nn.Conv2d):
    
    def __init__(self, type='B', *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        # Mask A) without center pixel
        # Mask B) with center pixel

        # 1 1 1 1 1 1 1
        # 1 1 1 1 1 1 1
        # 1 1 1 X 0 0 0
        # 0 0 0 0 0 0 0
        # 0 0 0 0 0 0 0

        self.mask = torch.ones_like(self.weight)
        _, _, height, width = self.weight.size()
        
        self.mask[:, :, height // 2, width // 2 + (1 if type=='B' else 0):] = 0
        self.mask[:, :, height // 2 + 1:] = 0

        if cuda:
            self.mask = self.mask.cuda()
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
            

class ResNetBlock(nn.Module):
    
    def __init__(self, num_filters=128):
        super(ResNetBlock, self).__init__()
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters//2, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d(in_channels=num_filters//2, out_channels=num_filters//2, kernel_size=3, padding=1, type='B'),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters//2, out_channels=num_filters, kernel_size=1)
        )
        
    def forward(self, x):
        return self.layers(x) + x


class PixelCNN(nn.Module):
    
    def __init__(self, num_layers=12, num_filters=128):
        super(PixelCNN, self).__init__()
        
        layers = [MaskedConv2d(in_channels=3,
                               out_channels=num_filters,
                               kernel_size=7,
                               padding=3, type='A')]
        
        for _ in range(num_layers):
            layers.append(ResNetBlock(num_filters=num_filters))
            
        layers.extend([
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=12, kernel_size=1)
        ])
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MaskedLinear(nn.Linear):
    
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)

        self.mask = torch.ones_like(self.weight)
        num_out, num_in = self.weight.size()
        
        self.mask[:num_out//2, :num_in//2] = 0

        if cuda:
            self.mask = self.mask.cuda()

    def forward(self, x):  
        self.weight.data *= self.mask
        return super(MaskedLinear, self).forward(x)


class MADE(nn.Module):

    def __init__(self):
        super(MADE, self).__init__()

        self.layers = nn.Sequential(
            MaskedLinear(in_features=6272, out_features=256),
            nn.ReLU(),
            MaskedLinear(in_features=256, out_features=256),
            nn.ReLU(),
            MaskedLinear(in_features=256, out_features=9408)
        )

    def forward(self, x):
        _batch_size, _channels, _classes, _h, _w = x.shape

        color = x[:, :1]
        colors = x[:, :2].view(_batch_size, -1)

        colors = self.layers(colors)

        colors = colors.view(_batch_size, 3, 4, _h, _w)

        x = colors
        x = x.permute(0, 3, 4, 1, 2)

        x = F.softmax(x, dim=-1)

        return x


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.pixelcnn = PixelCNN()
        self.made = MADE()
           
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    
    def optimize(self, X, take_step=True):
        y = torch.LongTensor(X.copy()).view(-1)

        if cuda:
            y = y.cuda()

        out = self(X)
        out = out.contiguous().view(-1, 4)

        loss = net.loss(out, y)
        
        if take_step:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        loss = loss.cpu().data.numpy().astype(np.float32)
        
        return loss
    

    def _sample(self, num_samples=1):
        samples = torch.Tensor(np.random.choice(4, size=(num_samples, 28, 28, 3)))

        for i in range(28):
            for j in range(28):
                out = self(samples)
                intensity = torch.distributions.Categorical(out).sample()
                samples[:, i, j, :] = intensity[:, i, j, :]

        return samples


    def sample(self, num_samples=5):
        self.eval()

        canvas = np.zeros((28, 28*num_samples, 3))
        
        samples = self._sample(num_samples)

        for i in range(num_samples):
            canvas[:, i*28:(i+1)*28] = rescale_rgb(samples[i])

        fig = plt.figure(figsize=(4, 4*num_samples))
        plt.axis('off')
        plt.imshow(canvas)
        plt.savefig('sample')
        plt.show()
        
        self.train()
        
    def forward(self, x):
        _batch_size, _h, _w, _channels = x.shape
        x = torch.Tensor(x).permute(0, 3, 1, 2)

        if cuda:
            x = x.cuda()

        x = self.pixelcnn(x)

        x = x.view(_batch_size, 3, 4, 28, 28)

        x = self.made(x)

        return x


cuda = torch.cuda.is_available()
do_train = False

print('CUDA has been enabled' if cuda else 'CUDA has been disabled')
print('\nInitializing network...')

net = Net()

if not do_train:
    net.load_state_dict(torch.load('net_full.pt', map_location=torch.device('cpu')))

if cuda:
    net = net.cuda()

epoch_loss = []
val_loss = []

if not do_train:
    print('Generating a sample...')
    net.sample()
    raise Exception('Not running on HPC, so we only do inference!')

print('Training...')

for epoch in range(num_epochs):

    batch_loss = []
    batch_loss_val = []
    
    for b in range(num_batches_train):
        
        X = train[get_slice(b, batch_size)]
        loss = net.optimize(X)
        batch_loss.append(loss)

    for b in range(num_batches_dev):

        X = dev[get_slice(b, batch_size)]
        loss = net.optimize(X, take_step=False)
        batch_loss_val.append(loss)

    epoch_loss.append(np.mean(batch_loss))
    val_loss.append(np.mean(batch_loss_val))

    torch.save(net.state_dict(), 'net.pt')

    print(f'Epoch {epoch+1}, train loss: {epoch_loss[-1]}, val loss: {val_loss[-1]}')

torch.save(net.state_dict(), 'net_full.pt')
