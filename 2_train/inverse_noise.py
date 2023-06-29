import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim

def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    # plt.figure(figsize=(8,5))
    plt.plot(x,y, color="gray", linewidth=2, alpha=0.8, label="Exact solution")
    
    plt.scatter(xp, oscillator(1.5, 15, torch.tensor(xp)), marker="^", color="black", label='gradint points')
    plt.scatter(x_data, y_data, marker="s", color="g", label='Training data')
    plt.scatter(0, 1, s=50, marker="*", c='r', label="init")
    
    plt.plot(x,yh, color="tab:blue", linewidth=3, alpha=0.8, label="Neural network prediction")
    
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

seed=1114
np.random.seed(seed)
torch.manual_seed(seed)


USE_GPU = True
dtype = torch.float64

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

# https://beltoforion.de/en/harmonic_oscillator/
def oscillator(delta, omega0, t):
    assert delta < omega0
    omega   = np.sqrt(omega0**2 - delta**2)
    phi     = np.arctan(-delta/omega)
    A       = 1/(2*np.cos(phi))
    return torch.exp(-delta * t)*2*A*torch.cos(phi+omega*t)

N_grid, delta, omega0 =100, 1.5, 15
mu_real, k_real = 2*delta, omega0**2        ## mass = 1

t_grid  = torch.linspace(0, 1, N_grid).view(-1,1)
x_grid  = oscillator(delta, omega0, t_grid).view(-1,1)

x_physics = torch.linspace(0,1,25).view(-1,1).requires_grad_(True)
x_physics = torch.Tensor.cuda(x_physics)

N_obs = 10
idx     = np.random.choice(np.arange(len(t_grid)), replace=0, size=N_obs)
t_obs   = t_grid[idx]
x_obs   = x_grid[idx]

NOISE = True
if NOISE:
    x_obs += torch.rand(x_obs.size())*0.1

figure = plt.figure(figsize=(5,3))
plt.plot(t_grid, x_grid, color='r', label='True $x(t)$')
plt.scatter(t_obs, x_obs, marker='o',label='Observed $x(t)$')
plt.legend(fontsize=8)

t_obs, x_obs             = t_obs.to(device).requires_grad_(True), x_obs.to(device)
t_grid, x_grid           = t_grid.to(device).requires_grad_(True), x_grid.to(device)
t_init, x_init, dx_init  = torch.zeros([1,1]).to(device).requires_grad_(True), torch.ones([1,1]).to(device), torch.zeros([1,1]).to(device)

k   = torch.ones(1).to(device).requires_grad_(True)
mu  = torch.ones(1).to(device).requires_grad_(True)

model = nn.Sequential(nn.Linear(1,64),
                      nn.Tanh(),
                      nn.Linear(64, 64),
                      nn.Tanh(),
                      nn.Linear(64, 64),
                      nn.Tanh(),
                      nn.Linear(64, 64),
                      nn.Tanh(),
                      nn.Linear(64, 1))
model = model.to(device)

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': k, 'lr':0.01}, {'params': mu, 'lr':0.5}], lr=0.001)

scheduler = None
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000], gamma=2)

def derivative(y, t) :
    return torch.autograd.grad(y, t, create_graph=True, grad_outputs=torch.ones(y.size()).to(device))[0]

a = torch.linspace(-15, 15, 30)
print(a*2)

history_loss, history_k, history_mu = [], [], []
history_MSE_real, history_MSE_obs = [], []

best_model=None
best_k, best_mu, best_loss = 0, 0, 100

# Train
model.train()

EPOCHS = 50000
weights = torch.sigmoid(torch.linspace(-15, 15, EPOCHS))*0.01
weights = weights.to(device)
files = []
for i in range(1, EPOCHS+1) :
    optimizer.zero_grad()

    output      = model(x_physics)                         # x(t)
    output_init = model(t_init)                         # x(t_init)
    output_obs  = model(t_obs)                          # x(t_obs)

    doutput     = derivative(output, x_physics)            # dx/dt(t)
    d2output    = derivative(doutput, x_physics)           # dx/dt(t)
    doutput_init= derivative(output_init, t_init)       # dx/dt(t_init)

    # Importance of ODS loss will be increase as epochs proceeds (For noisy case)
    loss_init   = loss_f(output_init, x_init) + loss_f(doutput_init, dx_init)                         # Loss for initial conditions
    loss_obs    = loss_f(output_obs, x_obs)                                                           # Loss for observed data
    loss_ge     = loss_f(d2output + mu*doutput + k*output, torch.zeros_like(doutput))*weights[i-1]   # Loss for ODE

    loss        = loss_init + loss_obs + loss_ge
    loss.backward()

    optimizer.step()
    if scheduler != None:
        scheduler.step()

    history_loss.append(loss.item())
    history_mu.append(mu.item())
    history_k.append(k.item())
    history_MSE_real.append(loss_f(x_grid, model(t_grid)).item())
    history_MSE_obs.append(loss_f(x_obs, model(t_obs)).item())

    if loss.item() < best_loss:
        best_model = model
        best_loss, best_mu, best_k = loss.item(), mu.item(), k.item()

    if not i % 1000 :
        print('EPOCH : %6d/%6d | MSE_real : %8.7f | MSE_obs : %8.7f | mu : %.3f | k : %.3f'\
              %(i, EPOCHS,  history_MSE_real[-1], history_MSE_obs[-1], mu.item(), k.item()))
        yh = model(t_grid.to(device)).detach()
        # print(yh)
        xp = t_grid.T
        
        file = "plots/pinn_%.8i.png"%(i)
        files.append(file)
        # print(t_grid.cpu().detach().numpy().T)
        a=t_grid.cpu().detach().numpy().T
        b=x_grid.cpu().detach().numpy().T
        c=t_obs.cpu().detach().numpy().T
        d=x_obs.cpu().detach().numpy().T
        e=yh.cpu().detach().numpy()

        plot_result(a,b,c,d,e,x_physics.cpu().detach().numpy().T)
        plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
        plt.cla()

print('Training Finished.')

save_gif_PIL("result/6_pinn_in_noise.gif", files, fps=20, loop=0)






# # View data
# figure = plt.figure(figsize=(5,3))
# plt.plot(t_grid.cpu().detach(), x_grid.cpu().detach(), color='k', linestyle='--', label='True $x(t)$')
# # plt.plot(t_grid.cpu().detach(), best_model(t_grid).cpu().detach(), color='r', label='PINN $x(t)$') # For best model
# plt.plot(t_grid.cpu().detach(), model(t_grid).cpu().detach(), color='r', label='PINN $x(t)$')        # For final model
# plt.scatter(t_obs.cpu().detach(), x_obs.cpu().detach(), marker='o',label='Observed $x(t)$')
# plt.legend(fontsize=8)
# plt.savefig('x_vs_t.png', dpi=300)

# figure = plt.figure(figsize=(15,5))
# ax1 = figure.add_subplot(1,2,1)
# ax1.set_yscale('log')
# ax1.plot(history_MSE_real)
# ax1.set_title('MSELoss btw PINN and Real Solution')
# ax1.set_xlabel('epoch')

# ax2 = figure.add_subplot(1,2,2)
# ax2.set_yscale('log')
# ax2.plot(history_MSE_obs)
# ax2.set_title('MSELoss btw PINN and Observed Data')
# ax2.set_xlabel('epoch')
# plt.savefig('MSE_loss.png', dpi=300)

# figure = plt.figure(figsize=(15,5))
# ax1 = figure.add_subplot(1,3,1)
# ax1.plot(history_mu)
# ax1.set_title('$\mu$')
# ax1.set_xlabel('epoch')
# ax1.hlines(mu_real, 0, EPOCHS, color='k',linestyle='--', label='True $\mu$')
# # ax1.hlines(best_mu, 0, EPOCHS, color='r', label='Best $\mu$') # For best model
# ax1.legend()


# ax2 = figure.add_subplot(1,3,2)
# ax2.plot(history_k)
# ax2.set_title('k')
# ax2.set_xlabel('epoch')
# ax2.hlines(k_real, 0, EPOCHS, color='k', linestyle='--', label='True k')
# # ax2.hlines(best_k, 0, EPOCHS, color='r', label='Best k') # For best model
# ax2.legend()


# ax3 = figure.add_subplot(1,3,3)
# ax3.plot(history_loss, label='Training Loss')
# ax3.set_yscale('log')
# ax3.legend()
# ax3.set_title('Training Loss in log scale', fontsize=13)
# ax3.set_xlabel('epoch', fontsize=13)
# plt.savefig('mu_k_loss.png', dpi=300)


