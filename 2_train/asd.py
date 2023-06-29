import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

seed=1114
np.random.seed(seed)
## device choice
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)

if device == 'cuda:2':
    torch.cuda.manual_seed_all(seed)
print("learning with",device,"\n")

def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    # plt.figure(figsize=(8,5))
    plt.plot(x,y, color="gray", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=1, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, marker="s", color="r", alpha=0.4, label='Training data')
    plt.legend(loc='lower right')
    plt.xlabel("Time", size=20)
    plt.ylabel("Displacement", size=20)
    plt.xlim(-0.05, 1.05)
    plt.xlim(-0., 1.)
    plt.ylim(-1.1, 1.1)
    # plt.text(.6,0.8,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    # plt.axis("off")

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

def pinn(hyper, generations, gif=False):
    learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden,d, w0 = hyper
    learingrate, number_of_epoch, nodes_per_hidden, number_of_hidden = float(learingrate), int(number_of_epoch), int(nodes_per_hidden), int(number_of_hidden) 
    # 1e-4 .        15000               32                  3
    def save_gif_PIL(outfile, files, fps=5, loop=0):
        "Helper function for saving GIFs"
        imgs = [Image.open(file) for file in files]
        imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    class FCN(nn.Module):
        "Defines a connected network"
        def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
            super().__init__()
            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                            nn.Linear(N_INPUT, N_HIDDEN),
                            activation()])
            ########################################################################
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(N_HIDDEN, N_HIDDEN),
                                activation()]) for _ in range(N_LAYERS-1)])
            ########################################################################
            # self.layer_hidden = nn.Sequential(
            #     nn.Linear(N_HIDDEN,N_HIDDEN),
            #     nn.ReLU())
            # self.hidden=nn.ModuleList()
            # for i in range(N_LAYERS):
            #     self.hidden.append(self.layer_hidden)
            ########################################################################
            self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        def forward(self, x):
            x = self.fcs(x)
            ########################################################################
            x = self.fch(x)
            ########################################################################
            # for layer in self.hidden:
            #     x = layer(x)
            ########################################################################
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
    
    k = torch.ones(1).to(device).requires_grad_(True)            # Parameter approximator
    mu = torch.ones(1).to(device).requires_grad_(True)
    
    x_data = torch.Tensor.cuda(x_data1)
    y_data = torch.Tensor.cuda(y_data1)

    ## pinn
    x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)# sample locations over the problem domain
    x_physics = torch.Tensor.cuda(x_physics)
    mu_t, k_t = 2*d, w0**2

    # os.system("mkdir plots")

    model = FCN(1,1,nodes_per_hidden,number_of_hidden).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()},\
                        {'params': k, 'lr':1e-1},\
                        {'params': mu, 'lr':1e-1}], lr=1e-3)
    files = []
    k_list, mu_list=[],[]
    for i in range(number_of_epoch):
        # print(f"{generations}gen\tepoch: \t{i}/{number_of_epoch}")
        optimizer.zero_grad()
        
        # compute the "data loss"
        yh = model(x_data)
        loss1 = criterion(yh,y_data)
        # loss1 = torch.mean((yh-y_data)**2).to(device)# use mean squared error
        
        # compute the "physics loss"
        yhp = model(x_physics)
        dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
        dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
        physics = dx2 + mu*dx + k*yhp# computes the residual of the 1D harmonic oscillator differential equation
        loss2 = (1e-4)*torch.mean(physics**2)
        
        # backpropagate joint loss
        loss = loss1 + loss2# add two loss terms together
        loss.backward()
        optimizer.step()
        
        k_list.append(k.item())
        mu_list.append(mu.item())
        
        # plot the result as training progresses
        if (i+1) %1000== 0: 
            
            print(i+1, loss1.item() , loss2.item())
            
            yh = model(x.to(device)).detach()
            xp = x_physics.detach()
            
            file = "plots/pinn_%.8i.png"%(i+1)
            files.append(file)
            
            
            if gif == True:
                plot_result(x,y,x_data,y_data,yh,xp)
                #plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            
            # if (i+1) % 6000 == 0: plt.show()
            # else: plt.close("all")
    if gif == True:
        save_gif_PIL("result/pinn.gif", files, fps=20, loop=0)

    return loss.item(), k_list, mu_list, mu_t, k_t, model, x_data1, y_data1

d, w0=1.5, 15
epoch=10000
hyper=[.0001, epoch, 60, 3,d, w0]
f_loss, ks, mus, mu_t, k_t, model, x_data1, y_data1 = pinn(hyper, generations=1, gif=False)

figure = plt.figure(figsize=(15,5))

xxx=range(epoch)

ax1 = figure.add_subplot(1,3,1)
ax1.plot(xxx,ks)
ax1.set_title('$\k$')
ax1.set_xlabel('epoch')
ax1.hlines(k_t, 0,epoch+100, label='True $\sigma$')

ax2 = figure.add_subplot(1,3,2)
ax2.plot(xxx, mus)
ax2.set_title('$\\mu$')
ax2.set_xlabel('epoch')
ax2.hlines(mu_t, 0,epoch+100,  label='True $\\rho$')

plt.savefig('lorenz2', dpi=100)
plt.show()
################################################################################
# ## # 통계 ->> 차이가 잘 안 보임.
# x = torch.linspace(0,1,500).view(-1,1)
# y = oscillator(d, w0, x).view(-1,1)

# z=0
# while z<5:
#     yh = model(x.to(device)).detach()
#     plot_result(x,y,x_data1,y_data1,yh.cpu())
#     z+=1
# plt.show()
################################################################################
xs = x_data1.numpy()#.T
ys = y_data1.numpy()#.T

x_test = np.array([5.])

def k_fun(X_1, X_2, gamma = .005):
    return np.exp(-gamma*(
        np.diag(np.matmul(X_1, X_1.T)).reshape([-1,1]) + np.diag(np.matmul(X_2, X_2.T)).reshape([1,-1]) - \
        2*np.matmul(X_1, X_2.T)))

tr_num = len(xs)
K = k_fun(xs.reshape([tr_num,-1]), xs.reshape([tr_num,-1]))
k = k_fun(x_test.reshape([1,-1]), xs.reshape([-1,1]))
# K

y_test = np.matmul(np.matmul(k, np.linalg.inv(K)), ys.reshape([-1,1]))
# print(y_test)

test_num = 50
xs_test = np.linspace(0, 1, num=test_num)
k = k_fun(xs_test.reshape([test_num,-1]), xs.reshape([tr_num,-1]))

ys_pred = np.matmul(np.matmul(k, np.linalg.inv(K)), ys.reshape([-1,1]))

k_test = k_fun(xs_test.reshape([test_num,-1]), xs_test.reshape([test_num,-1]))
sig_sq = .0005
var_test = sig_sq + np.diag(k_test) - np.diag(np.matmul(np.matmul(k, np.linalg.inv(K)), k.T))

plt.plot(xs_test, ys_pred)
plt.scatter(xs, ys)

plt.fill_between(
    xs_test.ravel(),
    ys_pred.ravel() - np.sqrt(var_test),
    ys_pred.ravel() + np.sqrt(var_test),
    alpha=0.5
)
plt.xlabel('x')
plt.ylabel('y')
plt.show()




################################################################################
# ## Gaussian kernel 2
def GaussianKernel(X1, X2, sig=1.):
    dist_sqs = np.sum(X1**2, axis=1).reshape([-1,1]) + \
        np.sum(X2**2, axis=1).reshape([1,-1]) - \
        2*np.matmul(X1, X2.T)
    K = np.exp(-.5*dist_sqs/sig**2)
    return K

# # Collection of functions
gp_sample_n = 50     # number of functions
xs = np.linspace(0, 1, gp_sample_n).reshape([-1,1])
sigma=1.5

# Posterior function generation
# print(x_data1)
# print(x_data1.numpy())
# print(x_data1.numpy().T)
# print(type(x_data1.numpy()))
tr_xs = x_data1.numpy()#.T
tr_ys = y_data1.numpy()#.T

k = GaussianKernel(tr_xs, xs, sigma)  # covariances
K = GaussianKernel(tr_xs, tr_xs, sigma)
invK = np.linalg.inv(K)

m_fun = np.matmul(np.matmul(k.T, invK), tr_ys).T[0]
k_fun = GaussianKernel(xs, xs, sigma) - np.matmul(np.matmul(k.T, invK), k)

ys = np.random.multivariate_normal(m_fun, k_fun, gp_sample_n)

# plt.scatter(tr_xs, tr_ys, s=1000)
for i in range(gp_sample_n):
    plt.plot(xs.T[0], ys[i], alpha=.3, c='k')
plt.scatter(tr_xs, tr_ys, s=30, c='k', zorder=5)
plt.show()
