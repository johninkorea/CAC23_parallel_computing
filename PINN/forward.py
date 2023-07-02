import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    # plt.figure(figsize=(8,5))
    plt.plot(x,y, color="gray", linewidth=2, alpha=0.8, label="Exact solution")
    
    plt.scatter(xp.cpu(), oscillator(1.5, 15, xp.cpu()), marker="^", color="black", label='gradint points')
    # plt.scatter(x_data.cpu(), y_data.cpu(), marker="s", color="g", label='Training data')
    plt.scatter(0, 1, s=50, marker="*", c='r', label="init")
    
    plt.plot(x,yh.cpu(), color="tab:blue", linewidth=3, alpha=0.8, label="Neural network prediction")
    
    plt.legend(loc='lower right')
    
    plt.xlabel("Time", size=20)
    plt.ylabel("Displacement", size=20)
    
    plt.xlim(-0.05, 1.05)
    plt.xlim(-0., 1.)
    plt.ylim(-1.1, 1.1)
    
    plt.text(.4,0.8,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    # plt.axis("off")

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

def oscillator(d, w0, x):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*x)
    sin = torch.sin(phi+w*x)
    exp = torch.exp(-d*x)
    y  = exp*2*A*cos
    return y

seed=1114
np.random.seed(seed)
## device choice
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)

if device == 'cuda:2':
    torch.cuda.manual_seed_all(seed)
print("learning with",device,"\n")
gif=True

d, w0=1.5, 15
mu, k = 2*d, w0**2
epoch=10000
hyper=[.0001, epoch, 60, 3,d, w0]

learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden,d, w0 = hyper
learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden = float(learingrate), int(number_of_epoch), int(nodes_per_hidden), int(number_of_hidden) 
# 1e-4 .        15000               32                  3
class FCN(nn.Module):
    "Defines a connected network"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


## create data to train
# d, w0 = 1.5, 15
# get the analytical solution over the full domain
x = torch.linspace(0,1,500).view(-1,1)
y = oscillator(d, w0, x).view(-1,1)

# slice out a small number of points from the LHS of the domain
index=np.random.choice(np.arange(len(x)), replace=0, size=10)
x_data1 = x[index]
y_data1 = y[index]


x_data = torch.Tensor.cuda(x_data1)
y_data = torch.Tensor.cuda(y_data1)

## pinn
x_physics = torch.linspace(0,1,25).view(-1,1).requires_grad_(True)# sample locations over the problem domain
# x_physics = torch.FloatTensor(20).uniform_(0,1).view(-1,1).requires_grad_(True)# sample locations over the problem domain
x_physics = torch.Tensor.cuda(x_physics)

# os.system("mkdir plots")

model = FCN(1,1,nodes_per_hidden,number_of_hidden).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

files = []
for i in range(number_of_epoch):
    # print(f"{generations}gen\tepoch: \t{i}/{number_of_epoch}")
    optimizer.zero_grad()
    
    # init loss
    y0=model(torch.zeros((1), device=device))
    loss0 = criterion(y0,torch.ones((1), device=device))

    # compute the "data loss"
    # yh = model(x_data)
    # loss1 = criterion(yh,y_data)
    # loss1 = torch.mean((yh-y_data)**2).to(device)# use mean squared error
    
    # compute the physics loss"
    yhp = model(x_physics)
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
    physics = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
    loss2 = (1e-4)*torch.mean(physics**2)
    
    # backpropagate joint loss
    loss = loss0 + torch.sigmoid(-15+torch.tensor(number_of_epoch)/30)*loss2 #+ loss3    # add two loss terms together
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) %100== 0: 
        
        print(i+1, loss0.item() , loss2.item())
        
        yh = model(x.to(device)).detach()
        xp = x_physics.detach()
        
        file = "plots/pinn_%.8i.png"%(i+1)
        files.append(file)
        
        
        if gif == True:
            plot_result(x,y,x_data,y_data,yh,xp)
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
            plt.cla()
        
        # if (i+1) % 6000 == 0: plt.show()
        # else: plt.close("all")
if gif == True:
    save_gif_PIL("result/pinn_fo.gif", files, fps=20, loop=0)






