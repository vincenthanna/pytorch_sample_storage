# -*- coding: utf-8 -*-

# %%
import torch
import math
from torchsummary import summary
import matplotlib.pyplot as plt

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

print(x.shape)

p = torch.tensor([1,2,3])
# x.shape = (2000,) 이므로 (2000,1)로 변경한다, pow()를 거치면 (2000, 3)이 됨
xx = x.unsqueeze(-1).pow(p)
print(xx.shape)

# 4(a,b,c,d)이여야 할 것 같지만 bias가 있어서 없는듯 싶다.
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=3, out_features=1),
    torch.nn.Flatten(start_dim=0, end_dim=1)
)

# select loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us.
lr = 1e-6
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

for t in range(2000):
    # make prediction
    y_pred = model(xx)

    # calculate loss with loss function
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # zero the gradient before running the backward pass.
    optimizer.zero_grad()

    # compute gradient of the loss with respect to all parameters of the model.
    loss.backward()
    
    # 아래와 같이 수동으로 각각의 parameter의 값을 갱신할 수 있지만, 단순히
    # lr * grad 값을 빼는 것보다 좋은 방법들이 나와있으므로 구현된 optimizer를
    # torch.optim에서 골라 사용하면 된다.
    '''
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    '''    
    # optimizer.step() makes an update to its parameters    
    optimizer.step()

ll = model[0]

# 3차방정식 형태로 보여준다.(a는 bias 임)
print(f'Result: y = {ll.bias.item()} + {ll.weight[:, 0].item()}x + {ll.weight[:, 1].item()}x^2 + {ll.weight[:, 2].item()}x^3')


# show input/target data
y_pred = model(xx) # predict with trained model
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(x, label="input")
ax.plot(y, label="target")
ax.plot(y_pred.detach().numpy(), label="pred")
ax.legend()
plt.show()




# %%
