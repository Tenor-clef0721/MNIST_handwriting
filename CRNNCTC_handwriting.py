import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# -------------------- 超参数 --------------------
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-4
epochs = 10
log_dir = './crnn_log'
data_dir = './MNIST_DATA'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 数据准备：两位数字拼接 --------------------
class TwoDigitMNIST(Dataset):
    """生成两位数字拼接图片，标签为两位数字字符串（如"12"）"""
    def __init__(self, mnist_data, transform=None, fixed_width=56):
        """
        mnist_data: 原始 MNIST 数据集 (PIL Image, label)
        transform: 对最终拼接图片的预处理（ToTensor, Normalize等）
        fixed_width: 固定输出宽度（两个28宽数字并排，无间隔）
        """
        self.mnist_data = mnist_data
        self.transform = transform
        self.fixed_width = fixed_width

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # 随机选择两个不同的样本（也可以允许相同）
        idx1 = random.randint(0, len(self.mnist_data)-1)
        idx2 = random.randint(0, len(self.mnist_data)-1)
        img1, label1 = self.mnist_data[idx1]
        img2, label2 = self.mnist_data[idx2]

        # 水平拼接
        total_width = img1.width + img2.width
        combined = Image.new('L', (total_width, 28), color=0)
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        # 缩放到固定宽度（保持高度28不变）
        if self.fixed_width is not None:
            combined = combined.resize((self.fixed_width, 28), Image.Resampling.LANCZOS)

        # 标签字符串，例如 "12"
        label_str = f"{label1}{label2}"

        if self.transform:
            combined = self.transform(combined)

        return combined, label_str

# 加载原始 MNIST（不进行变换，保留 PIL 图像）
mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
mnist_test  = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)

# 训练集和测试集的变换（ToTensor + 归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = TwoDigitMNIST(mnist_train, transform=transform, fixed_width=56)
test_dataset  = TwoDigitMNIST(mnist_test,  transform=transform, fixed_width=56)

# DataLoader 需要自定义 collate_fn 处理变长标签
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)        # (batch, 1, 28, 56)
    # 将标签字符串转为整数列表，并扁平化存储
    label_sequences = []
    label_lengths = []
    for s in labels:
        seq = [int(ch) for ch in s]            # 例如 "12" -> [1,2]
        label_sequences.extend(seq)
        label_lengths.append(len(seq))
    label_sequences = torch.tensor(label_sequences, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, label_sequences, label_lengths

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          collate_fn=collate_fn, drop_last=True)

# -------------------- 模型定义 --------------------
class CRNN(nn.Module):
    def __init__(self, num_classes=11, hidden_size=256):
        super().__init__()
        # CNN 特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14 -> 7

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # 高度 7 -> 3，宽度压缩比不同
        )
        # 自适应池化：将高度强制变为 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # RNN
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, W)
        features = self.cnn(x)                # (batch, 256, H', W')
        features = self.adaptive_pool(features)  # (batch, 256, 1, W')
        features = features.squeeze(2)        # (batch, 256, W')
        features = features.permute(0, 2, 1)  # (batch, W', 256)

        rnn_out, _ = self.rnn(features)       # (batch, W', hidden*2)
        output = self.fc(rnn_out)             # (batch, W', num_classes)
        output = F.log_softmax(output, dim=-1)
        return output

model = CRNN(num_classes=11).to(device)

# -------------------- 损失与优化器 --------------------
# CTC 损失，blank 索引为 10
ctc_loss = nn.CTCLoss(blank=10, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# -------------------- TensorBoard --------------------
writer = SummaryWriter(log_dir=log_dir)
dummy_input = torch.randn(1, 1, 28, 56).to(device)
writer.add_graph(model, dummy_input)

# -------------------- 辅助函数：记录参数 --------------------
def log_parameters(writer, model, step):
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'params/{name}', param.data, step)
            writer.add_scalar(f'params_mean/{name}', param.data.mean(), step)
            writer.add_scalar(f'params_std/{name}', param.data.std(), step)

# -------------------- 训练循环 --------------------
print("Start training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(images)                        # (batch, seq_len, 11)
        # 输入序列长度（所有样本相同，因为图片宽度固定为56，CNN输出序列长度固定）
        input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)

        # CTC 损失要求 log_probs 格式为 (seq_len, batch, num_classes)
        loss = ctc_loss(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1:2d} | Batch {batch_idx:4d} | Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('loss/train', avg_loss, epoch)

    # 每个 epoch 结束后在测试集上评估
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets, target_lengths in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long).to(device)
            loss = ctc_loss(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            test_loss += loss.item()

            # 解码预测结果
            preds = outputs.argmax(dim=-1)               # (batch, seq_len)
            # 将每个样本的预测转换为字符串（去除连续重复和blank）
            for i in range(preds.size(0)):
                pred_seq = preds[i].cpu().numpy()
                # 贪婪解码
                decoded = []
                prev = -1
                for token in pred_seq:
                    if token != prev and token != 10:   # 10 是 blank
                        decoded.append(str(token))
                    prev = token
                pred_str = ''.join(decoded)
                # 获取真实标签
                start = sum(target_lengths[:i]) if i>0 else 0
                end = start + target_lengths[i]
                true_seq = targets[start:end].cpu().numpy()
                true_str = ''.join([str(t) for t in true_seq])
                if pred_str == true_str:
                    correct += 1
                total += 1

    test_loss /= len(test_loader)
    test_acc = correct / total
    writer.add_scalar('loss/test', test_loss, epoch)
    writer.add_scalar('accuracy/test', test_acc, epoch)

    print(f'Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

print("Training finished.")
writer.close()
torch.save(model.state_dict(), 'crnn_twodigit.pth')
print("Model saved as crnn_twodigit.pth")