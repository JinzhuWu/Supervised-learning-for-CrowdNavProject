import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import os

BATCH_SIZE = 5
train_input = []
train_target = []
test_input = []
test_target = []
learning_rate = 0.0001 
filename = 'record_new.txt'
with open(filename,'r') as file:
	count = 0
	while True:
		count+=1
		if count > 13000:
			break
		elif count < 10000:
			input_data = file.readline().strip('\n[] ').split(',')
			target_data = file.readline().strip('\n[] ').split(',')
			input_data = [float(x) for x in input_data]
			target_data = [float(x) for x in target_data]
			train_input.append(input_data)
			train_target.append(target_data)
		else:
			input_data = file.readline().strip('\n[] ').split(',')
			target_data = file.readline().strip('\n[] ').split(',')
			input_data = [float(x) for x in input_data]
			target_data = [float(x) for x in target_data]
			test_input.append(input_data)
			test_target.append(target_data)

x = torch.tensor(train_input,dtype = torch.float)
y = torch.tensor(train_target,dtype = torch.float)
test_x = Variable(torch.tensor(test_input,dtype = torch.float))
test_y = Variable(torch.tensor(test_target,dtype = torch.float))


torch_dataset = Data.TensorDataset(x, y)


loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True,
    num_workers=2 #多线程
)
 
class Net(torch.nn.Module):
    def __init__(self, netinput, hidden, output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(netinput, hidden) #隐藏层的线性输出
        self.out = torch.nn.Linear(hidden, output) #输出层线性输出
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x
 

def train_model(learning_rate):
	# input_dim = 13
	# hidden_dim = 150,100,100
	# output_dim = 36
    net = torch.nn.Sequential(
        torch.nn.Linear(13,150), 
        torch.nn.ReLU(), #隐藏层非线性化
        torch.nn.Linear(150,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,100),
        torch.nn.ReLU(),
        torch.nn.Linear(100,36) 
    )
 	# 优化算法
 	# optimizer = torch.optim.SGD( net.parameters(), lr=learning_rate )
    # optimizer = torch.optim.SGD( net.parameters(), lr=learning_rate, momentum=0.9 )
    # optimizer = torch.optim.RMSprop(  net.parameters(), lr=learning_rate, alpha=0.9 )
    optimizer = torch.optim.Adam( net.parameters(), lr=learning_rate, betas=(0.9,0.99) )
 
    # 损失函数
    loss_func = torch.nn.MSELoss()

    epoch_num = 2000
    eval_interval = 100
    for epoch in range(epoch_num):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = net(Variable(batch_x))
            loss = loss_func(out, Variable(batch_y))
 
            optimizer.zero_grad() #清除上次迭代的更新梯度
            loss.backward() 
            optimizer.step() #更新权重
 
        if epoch%eval_interval==0:
            entire_out = net(x) #测试整个训练集
            pred = torch.max(entire_out,1)[1] #torch.max用法 torch.max(a,1) 返回每一行中最大值的那个元素，troch.max()[1]，只返回最大值的每个索引
            label = torch.max(y,1)[1]
            pred_y = pred.data.numpy()
            label_y = label.data.numpy()
            success = 0
            for i in range(len(pred_y)):
            	if pred_y[i] == label_y[i]:
                	success += 1
            accuracy = success/len(pred_y)
            print("第 %d 个epoch，准确率为 %.2f"%(epoch+1, accuracy))

 
    # 保存模型结构和参数
    torch.save(net, 'net.pkl')
    # 只保存模型参数
    # torch.save(net.state_dict(), 'net_param.pkl')


def load_model(test_x,test_y):
    net = torch.load('net.pkl')
    print('成功导入模型！')
    entire_out = net(test_x) #测试整个训练集
    pred = torch.max(entire_out,1)[1]
    label = torch.max(test_y,1)[1]
    pred_y = pred.data.numpy()
    label_y = label.data.numpy()
    success = 0
    for i in range(len(pred_y)):
    	if pred_y[i] == label_y[i]:
    		success += 1
    accuracy = success/len(pred_y)
    print("测试样本数为 %d ，测试准确率为 %.2f"%(len(pred_y), accuracy))

 
if __name__ == '__main__':
    if os.path.exists('net.pkl'):
        os.remove('net.pkl')
    train_model(learning_rate)
    load_model(test_x,test_y)
