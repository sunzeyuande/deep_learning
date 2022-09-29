import torch
import pickle
import gzip
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.relu(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(100, self.hidden_size)


with gzip.open('mnist/mnist.pkl_3.gz', "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")  # 加载数据集，划分训练集验证集及其对应标签

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))  # 将数据转为张量格式
n, c = x_train.shape  # 获得张量形状
bs = 100  # batch_size大小
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)  # 划分训练集并打乱顺序
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs)  # 划分验证集

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu加速
lr = 0.001
epochs = 25
n_input = 28
n_output = 10
n_hidden = 100
rnn = RNN(n_input, n_hidden, n_output).to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(rnn.parameters())  # adam优化器
loss_plot = []
rnn.load_state_dict(torch.load('./mnist/mnist.pkl'))

# train_start = time.time()
# for epoch in range(epochs):
#     lst = []
#     total_loss = 0
#     plot_loss = 0
#     count = 0
#     train_correct = 0
#
#     for xb, yb in train_dl:
#         hidden = rnn.initHidden().to(device)
#         xb = xb.to(device)  # 将张量部署到gpu
#         xb = xb.view(100, 28, 28)
#         yb = yb.to(device)
#         for i in range(28):
#             output, hidden = rnn(xb[:, i, :], hidden)
#
#         loss = criterion(output, yb)
#         loss.backward()  # 网络参数更新
#         optimizer.step()  # 优化器梯度更新
#         optimizer.zero_grad()  # 优化器梯度清零
#         preds = torch.argmax(output, dim=1)  # 识别数字是几
#         train_correct += sum(preds == yb)
#
#         count += 1
#         total_loss += loss.item()
#         plot_loss += loss.item()
#         if count % 10 == 0:
#             # print('train_loss:', plot_loss / 10)
#             loss_plot.append(plot_loss/10)
#             plot_loss = 0
#     train_acc = train_correct.item()/50000  # 计算acc
#     print('epoch:', epoch+1, 'train_loss:', total_loss / 500, 'train_acc:', train_acc)
#
#     plt.figure()  # 每轮训练后绘图
#     plt.plot(loss_plot)
#     plt.title('loss of epoch'+str(epoch+1))
#     plt.xlabel('number of 10batches')
#     plt.ylabel('loss')
#     plt.show()
#
# train_finish = time.time()
# print('训练时间(s)：', train_finish-train_start)  # 计算训练时间
# torch.save(rnn.state_dict(), './mnist/mnist.pkl')  # 保存网络参数

with torch.no_grad():  # 网络权重不更新
    lst = []
    loss = 0
    test_correct = 0
    for xb, yb in valid_dl:
        hidden = rnn.initHidden().to(device)
        xb = xb.to(device)  # 将张量部署到gpu
        xb = xb.view(100, 28, 28)
        yb = yb.to(device)

        for i in range(28):
            output, hidden = rnn(xb[:, i, :], hidden)
        loss_pre = criterion(output, yb).item()
        loss += loss_pre

        preds = torch.argmax(output, dim=1)  # 识别数字是几

        lst.append(yb.cpu().numpy())
        lst.append(preds.cpu().numpy())
        test_correct += (sum(preds == yb)).item()
    test_acc = test_correct/10000  # 计算acc
    test_loss = loss/100
    print('实际数字：', lst[-2][0:30])
    print('识别数字：', lst[-1][0:30])
    print('test_loss:', test_loss, 'test_acc:', test_acc)
