import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
import matplotlib as plt

num_epochs=100
batch_size=100
learning_rate=0.001

train_dataset=datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=datasets.MNIST(root='data',train=False,transform=transforms.ToTensor(),download=True)


train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        #只有一个隐藏层
        #config
        self.layer1=nn.Linear(784,300)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(300,10)

    def forward(self,x):
        x=x.reshape(-1,28*28)
        x=self.layer1(x)
        x=self.relu(x)
        y=self.layer2(x)

        return y

mlp=MLP()
#交叉熵实际上表示的是惊讶程度，我们只需要让惊讶程度最小就可以了
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(mlp.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        outputs=mlp(images)
        loss=loss_func(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print('epoch==',epoch,'i==',i+1,'loss==',loss)


#测试模型
mlp.eval()
correct=0
total=0

for images,labels in test_loader:
    outputs=mlp (images)
    _,predicted=torch.max(outputs,dim=1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum().item()

print('测试准确率：',100*correct/total)










