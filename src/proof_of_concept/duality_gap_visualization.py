import torch 
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import math 
from itertools import chain 


"""
calculate the best case for each
then optimize given the best case of the other

"""


def f(x,y):
    return 3*x**2 - y**2 + 4*x*y

#def f(x,y):
#    return 3*x**2 + y**2 + 4*x*y

def gradient_descent_ascent(x,y,f):
    return f(x,y.detach()) - f(x.detach(),y)

def duality_gap_objective(x,y,f,k_steps=40,surrogate_lr=0.01):
    x_surrogate = nn.Parameter(torch.tensor([x.item()]))
    y_surrogate = nn.Parameter(torch.tensor([y.item()]))
    
    optimizer_x_worst = optim.SGD([x_surrogate],lr=surrogate_lr)
    optimizer_y_worst = optim.SGD([y_surrogate],lr=surrogate_lr)
    
    #calcute best case
    for _ in range(k_steps):
        optimizer_x_worst.zero_grad()
        optimizer_y_worst.zero_grad()
        loss_x = f(x_surrogate,y.detach()) 
        loss_y = -f(x.detach(),y_surrogate) 
        loss_x.backward()
        loss_y.backward()
        optimizer_x_worst.step()
        optimizer_y_worst.step()
    #duality gap
    duality_gap_loss = f(x,y_surrogate.detach())-f(x_surrogate.detach(),y)  #given best case of the other maximize your own results
    return duality_gap_loss


if __name__ == "__main__":
    #hyperparam
    lr = 0.1

    #init
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    x = torch.linspace(-5,5,1000)
    y = torch.linspace(-5,5,1000)
    X,Y = torch.meshgrid(x,y)
    Z = f(X,Y)

    x_plot = []
    y_plot = []
    z_plot = []

    x_coordinate = nn.Parameter(torch.tensor([3.0]))
    y_coordinate = nn.Parameter(torch.tensor([3.0]))
    #x_coordinate = nn.Parameter(torch.randn(1)*3)
    #y_coordinate = nn.Parameter(torch.randn(1)*3)
    optimizer = optim.Adam((x_coordinate,y_coordinate),lr=lr)

    plt.ion()
    for _ in range(1000):

        #visualization
        ax.cla()
        ax.plot_surface(X.numpy(), Y.numpy(),Z.numpy() ,cmap='viridis', edgecolor='none',alpha=0.8)
        x_pos = x_coordinate.detach().item()
        y_pos = y_coordinate.detach().item()
        z_pos = f(x_coordinate,y_coordinate).detach().item()
        x_plot.append(x_pos)
        y_plot.append(y_pos)
        z_plot.append(z_pos)
        ax.scatter(x_pos,y_pos,z_pos,color="red")
        ax.plot(x_plot,y_plot,z_plot)
        plt.pause(0.1)

        #update 
        optimizer.zero_grad()
        loss = duality_gap_objective(x_coordinate,y_coordinate,f)
        loss.backward()
        optimizer.step()



    plt.show()