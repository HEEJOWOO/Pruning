import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1,6,3)
		self.conv2 = nn.Conv2d(6,16,3)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)
	def forward(self,x):
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(-1, int(x.nelement() / x.shape[0]))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

model = LeNet().to(device)

module = model.conv1
#print(list(module.named_parameters()))
#print(list(module.named_buffers()))

#pruning
print("-----------------------------before weight pruning------------------------")
print(module.weight)
prune.random_unstructured(module, name = "weight", amount = 0.3)
#print(list(module.named_parameters()))
#print(list(module.named_buffers()))
print("-----------------------------after weight pruning------------------------")
print(module.weight)
print(module._forward_pre_hooks)
print("-----------------------------before bias  pruning------------------------")
print(module.bias)
prune.l1_unstructured(module, name = "bias", amount=3)
print("-----------------------------after bias  pruning------------------------")
print(module.bias)
#print(list(module.named_parameters()))
#print(list(module.named_buffers()))
#print(module.bias)
print(module._forward_pre_hooks)

prune.ln_structured(module, name = "weight", amount=0.5,n=2,dim=0)
print(module.weight)

for hook in module._forward_pre_hooks.values():
	if hook._tensor_name == "weight":
		break

print(list(hook))
print(model.state_dict().keys())

#prune remove
print("-----------------------------before pruning remove------------------------")
print(list(module.named_parameters()))
prune.remove(module,'weight')
print("-----------------------------after pruning remove------------------------")
print(list(module.named_parameters()))
