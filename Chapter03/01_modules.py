# 本程序演示了如何使用torch构建deep neural network

import torch
import torch.nn as nn

# 自定义NN类, 继承自torch.nn.Module
# 核心是重载forward()函数
class OurModule(nn.Module):
    # 定义一个串行NN
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    # 重载forward()
    def forward(self, x):
        return self.pipe(x)

# 主程序开始
if __name__ == "__main__":
    # 生成NN对象
    net = OurModule(num_inputs=2, num_classes=3)
    # 打印网络结构
    print(net)
    # 定义输入
    v = torch.FloatTensor([[2, 3]])
    # 网络预测结果
    out = net(v)
    # 打印输出
    print(out)
    # 判断是否存在gpu
    print("Cuda's availability is %s" % torch.cuda.is_available())
    # 将输出转到gpu中
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to('cuda'))
