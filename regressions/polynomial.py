# -*- coding: utf-8 -8-

"""
1차 방정식값을 sin(x)으로 변환하는 3차방정식의 계수를 pytorch를 
사용해서 구한다.
"""

# %%
import torch
import math
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)



# %%


"""
다항식의 계수로 사용할 파라메터를 tensor로 생성
y = a + bx + cx**2 + dx**3
"""
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3

    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
    
print(f'Result: y = {a.item()}, + {b.item()}x + {c.item()}x^2 + {d.item()}^3')


# %%
y_pred = a + b*x + c*x**2 + d*x**3

# show input/target data
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(x, label="input")
ax.plot(y, label="target")
ax.plot(y_pred.detach().numpy(), label="pred")
ax.legend()
plt.show()
# %%
