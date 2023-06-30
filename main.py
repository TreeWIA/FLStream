import copy
import math

import numpy as np
import arff
import torch
import torch.nn.functional as F
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import itertools
import pandas as pd
import random
from torch.utils.data import TensorDataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_file):
        super(MyDataset, self).__init__()

        # Check the file extension
        if data_file.endswith('.csv'):
            self._load_csv(data_file)
        elif data_file.endswith('.arff'):
            self._load_arff(data_file)
        else:
            raise ValueError("Unsupported file format.")

    def _load_csv(self, data_file):
        # Load data from CSV file
        with open(data_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            data = list(csv_reader)

        # Extract features and labels from the loaded data
        self.features = []
        self.labels = []
        for line in data:
            self.features.append([float(f) for f in line[:-1]])
            self.labels.append(float(line[-1]))
        self.features = torch.tensor(self.features)
        self.labels = torch.tensor(self.labels)

    def _load_arff(self, data_file):
        # Load data from ARFF file
        with open(data_file, 'r') as f:
            lines = f.readlines()

        data_start = False
        data_lines = []
        for line in lines:
            if not data_start:
                if line.startswith('@data'):
                    data_start = True
            else:
                if not line.startswith('@'):
                    data_lines.append(line.strip())

        # Process data lines to extract features and labels
        data = [line.split(',') for line in data_lines]
        self.features = torch.tensor([list(map(float, row[:-1])) for row in data], dtype=torch.float32)
        self.labels = torch.tensor([float(row[-1]) for row in data], dtype=torch.float32)

    def split_dataset(self, num_splits):
        # Calculate the number of samples in each split
        split_size = len(self.features) // num_splits

        # Split the dataset into num_splits parts
        split_features = [[] for _ in range(num_splits)]
        split_labels = [[] for _ in range(num_splits)]
        for i in range(len(self.features)):
            split_index = i % num_splits

            split_features[split_index].append(self.features[i])
            split_labels[split_index].append(self.labels[i])

        return split_features, split_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


#  MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# client update
def client_update(model, optimizer, data_stream, train_losses, train_accs):
    model.train()
    correct = 0
    data, target = data_stream
    target = target.long()
    # data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    train_losses.append(loss.item())
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    train_accs.append(correct / len(data))
    loss.backward()
    optimizer.step()

    # print('\t\tTrain: Average loss {:.3f}, Average accuracy {:.3f}'.format(
    #     sum(train_losses) / len(train_losses),
    #     sum(train_accs) / len(train_accs)
    # ))


# fedavg
def server_aggregate(global_model, local_models):
    for global_param, local_param_list in zip(global_model.parameters(),
                                              zip(*[local_model.parameters() for local_model in local_models])):
        global_param.data = torch.mean(torch.stack(local_param_list), dim=0)


# test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = test_loader
        target = target.long()
        # data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target = target.cpu()
        # pred = pred.cpu()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    # print(f'\n\t\tTest: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data)} F1-score: {test_f1:.4f}'
    #       f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy, test_f1


def test1(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    modified_dataset = []  # save date after predict
    with torch.no_grad():
        data, target = test_loader
        target = target.long()
        # data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        is_correct = pred.eq(target.view_as(pred)).float()  # correct 1，false 0
        modified_dataset.append((data, is_correct))
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target = target.cpu()
        # pred = pred.cpu()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    # print(f'\n\t\tTest: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data)} F1-score: {test_f1:.4f}'
    #       f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy, test_f1, modified_dataset

# initial parameter
drift_type = 'gradual_data_L'
csv_name = f"{drift_type}"
gpu = 0
device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')
print(device)

criterion = nn.CrossEntropyLoss()

num_clients = 10
batch_size = 100
lr = 0.005
num_epochs = 10
testnum = 'covt-10'

# select dataset
train_datasets = []
train_dataset = MyDataset(f'data\Real-world datasets\covType\covtypeNorm.arff')
split_features, split_labels = train_dataset.split_dataset(num_clients)

for i in range(num_clients):
    train_dataset = MyDataset(f'data\Real-world datasets\covType\covtypetest.arff')
    train_dataset.features = split_features[i]
    train_dataset.labels = split_labels[i]
    train_datasets.append(train_dataset)
# train_datasets = []
# for i in range(num_clients):
#     # if i == num_clients-1:
#     #     csv_name = f"data/rotation/abrupt_data_H_9.csv"
#     #     train_dataset = MyDataset(csv_name)
#     #     train_datasets.append(train_dataset)
#     # elif i == num_clients-2:
#     #     csv_name = f"data/rotation/abrupt_data_H_8.csv"
#     #     train_dataset = MyDataset(csv_name)
#     #     train_datasets.append(train_dataset)
#     # elif i == num_clients-3:
#     #     csv_name = f"data/rotation/abrupt_data_H_7.csv"
#     #     train_dataset = MyDataset(csv_name)
#     #     train_datasets.append(train_dataset)
#     # elif i == num_clients-4:
#     #     csv_name = f"data/rotation/abrupt_data_H_6.csv"
#     #     train_dataset = MyDataset(csv_name)
#     #     train_datasets.append(train_dataset)
#     # elif i == num_clients-5:
#     #     csv_name = f"data/rotation/abrupt_data_H_5.csv"
#     #     train_dataset = MyDataset(csv_name)
#     #     train_datasets.append(train_dataset)
#     # else:
#         csv_name = f"{drift_type}"
#         csv_name = f"data/rotation/{csv_name}_{i}.csv"
#         train_dataset = MyDataset(csv_name)
#         train_datasets.append(train_dataset)

min_length = 1000000
shortest_dataset = None
for dataset in train_datasets:
    if len(dataset) < min_length:
        min_length = len(dataset)
        shortest_dataset = dataset

print(f"The shortest dataset has length {min_length}.")
globle_train_dataset = shortest_dataset

data_streams = []
for i in range(num_clients):
    data_stream = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=False)
    data_streams.append((data_stream))

# for i in range(num_clients):
#     data_streams[i] = iter(data_streams[i])
# # data_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# global_model = copy.deepcopy(model)

client_test_f1ss = []
client_test_losses = []
client_test_accs = []
global_test_accs = []
global_test_losses = []
global_test_f1s = []
pre_global_datas = [[] for _ in range(num_epochs)]
pre_client_datas = [[[] for _ in range(num_clients)] for _ in range(num_epochs)]
pre_client_data = [[[[] for _ in range(2)] for _ in range(num_clients)] for _ in range(num_epochs)]

for j in range(num_epochs):
    model = MLP(54,1000,8)  # 54,1000,8      8,1000,2
    model = model.to(device)
    print(j)
    global_data_stream = iter(DataLoader(globle_train_dataset, batch_size=batch_size))
    global_model = copy.deepcopy(model)
    data_streams = []
    client_test_f1s = [[] for _ in range(num_clients)]
    client_test_loss = [[] for _ in range(num_clients)]
    client_test_acc = [[] for _ in range(num_clients)]
    global_test_f1 = []
    global_test_acc = []
    global_test_loss = []
    global_data_batch = []

    for i in range(num_clients):
        data_stream = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=False)
        data_streams.append((data_stream))

    for i in range(len(global_data_stream)):
        # print('\ntime stamp:',i)
        local_models = []
        local_train_losses = []
        local_train_accs = []
        global_data = []
        for client_idx in range(num_clients):
            # print('\n\tclient {}'.format(client_idx))
            data_streams[client_idx] = iter(data_streams[client_idx])
            client_data = next(data_streams[client_idx])
            if len(client_data[1]) >= 10:
                batch_size_10_percent = int(len(client_data[1]) * 0.1)
                selected_data = []
                # print(len(client_data[1]))
                selected_data = random.sample(list(zip(*client_data)), batch_size_10_percent)
                selected_data = list(zip(*selected_data))
                tensor_list = [tensor.unsqueeze(0) for tensor in selected_data[0]]
                combined_tensor = torch.cat(tensor_list, dim=0)
                scalar_tensor = torch.stack(selected_data[1], dim=0)
                # combine data,label
                selected_data_result = [combined_tensor, scalar_tensor]
                selected_data_result = [t.to(device) for t in selected_data_result]
                selected_data = [torch.stack(d).to(device) for d in selected_data]
                if global_data == []:
                    global_data = selected_data_result
                else:
                    global_data = [torch.cat([global_data[0], selected_data_result[0]], dim=0),
                                   torch.cat([global_data[1], selected_data_result[1]], dim=0)]

                if global_data_batch == []:
                    global_data_batch = selected_data_result
                else:
                    global_data_batch = [torch.cat([global_data_batch[0], selected_data_result[0]], dim=0),
                                         torch.cat([global_data_batch[1], selected_data_result[1]], dim=0)]

            client_data = [d.to(device) for d in client_data]

            # client model test-then-train
            local_model = copy.deepcopy(model)
            local_model.to(device)
            local_model.load_state_dict(global_model.state_dict())

            test_loss, test_acc, test_f1, client_test_data = test1(local_model, device, client_data)

            cpu_data = []
            for tuple_item in client_test_data:
                cpu_tuple = tuple(tensor.cpu() for tensor in tuple_item)
                cpu_data.append(cpu_tuple)
            pre_client_datas[j][client_idx].extend(cpu_data)

            client_test_acc[client_idx].append(test_acc)
            client_test_loss[client_idx].append(test_loss)
            client_test_f1s[client_idx].append(test_f1)
            # client train
            client_optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            client_update(local_model, client_optimizer, client_data, local_train_losses, local_train_accs)
            local_models.append(local_model)

        # global data
        global_loss, global_acc, global_f1, pre_global_datas[j] = test1(global_model, device, global_data_batch)
        global_loss, global_acc, global_f1 = test(global_model, device, global_data)
        # parameters aggregation
        server_aggregate(global_model, local_models)
        global_test_acc.append(global_acc)
        global_test_loss.append(global_loss)
        global_test_f1.append(global_f1)

    if len(client_test_accs) == 0:
        client_test_accs = client_test_acc
    else:
        client_test_accs = np.array(client_test_accs)
        client_test_acc = np.array(client_test_acc)
        client_test_accs = client_test_accs + client_test_acc

    if len(client_test_losses) == 0:
        client_test_losses = client_test_loss
    else:
        client_test_losses = np.array(client_test_losses)
        client_test_loss = np.array(client_test_loss)
        client_test_losses = client_test_losses + client_test_loss

    if len(client_test_f1ss) == 0:
        client_test_f1ss = client_test_f1s
    else:
        client_test_f1ss = np.array(client_test_f1ss)
        client_test_f1s = np.array(client_test_f1s)
        client_test_f1ss = client_test_f1ss + client_test_f1s

    if len(global_test_f1s) == 0:
        global_test_f1s = global_test_f1
    else:
        global_test_f1s = np.array(global_test_f1s)
        global_test_f1 = np.array(global_test_f1)
        global_test_f1s = global_test_f1s + global_test_f1
        global_test_f1s = global_test_f1s.tolist()

    if len(global_test_accs) == 0:
        global_test_accs = global_test_acc
    else:
        global_test_accs = np.array(global_test_accs)
        global_test_acc = np.array(global_test_acc)
        global_test_accs = global_test_accs + global_test_acc
        global_test_accs = global_test_accs.tolist()

    if len(global_test_losses) == 0:
        global_test_losses = global_test_loss
    else:
        global_test_losses = np.array(global_test_losses)
        global_test_loss = np.array(global_test_loss)
        global_test_losses = global_test_losses + global_test_loss
        global_test_losses = global_test_losses.tolist()

# save client data
for j in range(num_epochs):
    for i in range(num_clients):

        concatenated_tensors_data = []
        concatenated_tensors_label = []
        for k in range(math.ceil(min_length / batch_size)):

            concatenated_tensors_data.append(pre_client_datas[j][i][k][0])
            concatenated_tensors_label.extend(pre_client_datas[j][i][k][1])

        concatenated_tensor_data = torch.cat(concatenated_tensors_data, dim=0)
        concatenated_tensor_label = torch.cat(concatenated_tensors_label, dim=0)

        pre_client_data[j][i][0] = concatenated_tensor_data
        pre_client_data[j][i][1] = concatenated_tensor_label


print(pre_client_data[0][1][0])
print(pre_client_data[0][1][1])
print(len(pre_client_data[0][0][1]))

for j in range(num_epochs):
    for i in range(num_clients):
        df = pd.DataFrame(pre_client_data[j][i][0].numpy())
        df['Label'] = pre_client_data[j][i][1].numpy()
        df.to_csv(f'client{i}_{testnum}_test_{j}.csv', index=False)

for j in range(num_epochs):
    pre_global_datas[j] = [(t[0].cpu(), t[1].cpu()) for t in pre_global_datas[j]]
    df = pd.DataFrame(pre_global_datas[j][0][0])
    df['Label'] = pre_global_datas[j][0][1]
    df.to_csv(f'global_{testnum}_test_{j}.csv', index=False)

client_test_accs = client_test_accs / num_epochs
client_test_losses = client_test_losses / num_epochs
client_test_f1ss = client_test_f1ss / num_epochs

global_test_accs = np.array(global_test_accs)
global_test_losses = np.array(global_test_losses)
global_test_f1s = np.array(global_test_f1s)
global_test_accs = global_test_accs / num_epochs
global_test_losses = global_test_losses / num_epochs
global_test_f1ss = global_test_f1s / num_epochs

for i, avg_acc in enumerate(client_test_accs):
    plt.plot(avg_acc, label=f"Client {i + 1}")

plt.xlabel('Time stamp')
plt.ylabel('Test accs')
plt.title('Test accs of Clients')
plt.ylim(0, 101)
plt.legend()
plt.savefig(f"new-results/{testnum}_accs.png")
plt.show()

for i, avg_loss in enumerate(client_test_losses):
    plt.plot(avg_loss, label=f"Client {i + 1}")

plt.xlabel('Time stamp')
plt.ylabel('Test loss')
plt.title('Test loss of Clients')
plt.ylim(0, 0.03)
plt.legend()
plt.savefig(f"new-results/{testnum}_losses.png")
plt.show()

for i, avg_f1s in enumerate(client_test_f1ss):
    plt.plot(avg_f1s, label=f"Client {i + 1}")
plt.xlabel('Time stamp')
plt.ylabel('Test f1s')
plt.title('Test f1s of Clients')
plt.ylim(0, 1.1)
plt.legend()
plt.savefig(f"new-results/{testnum}_f1s.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(global_test_losses)), global_test_losses)
plt.xlabel('Time stamp')
plt.ylabel('Test Loss')
plt.title('Global Test Loss')
plt.savefig(f"new-results/global_{testnum}_loss.png")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(len(global_test_accs)), global_test_accs)
plt.xlabel('Time stamp')
plt.ylabel('Test Accuracy')
plt.ylim(0, 101)
plt.title('Global Test Accuracy')
plt.savefig(f"new-results/global_{testnum}_acc.png")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(global_test_f1s)), global_test_f1s)
plt.xlabel('Time stamp')
plt.ylabel('Test F1 Score')
plt.ylim(0, 1.1)
plt.title('Global Test F1 Score')
plt.savefig(f"new-results/global_{testnum}_f1.png")
plt.grid(True)
plt.show()
