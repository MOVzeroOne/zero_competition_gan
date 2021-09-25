import torch 
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import math 
from itertools import chain 



def f(x,y):
    return 3*x**2 - y**2 + 4*x*y


#def f(x,y):
#    return 3*x**2 + y**2 + 4*x*y

def gradient_descent_ascent(x,y,f):
    return f(x,y.detach()) - f(x.detach(),y)

if __name__ == "__main__":
    #hyperparam
    lr = 0.1

    #init
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #x = torch.linspace(-0.5,0.5,1000)
    #y = torch.linspace(-0.5,0.5,1000)
    x = torch.linspace(-5,5,1000)
    y = torch.linspace(-5,5,1000)
    X,Y = torch.meshgrid(x,y)
    Z = f(X,Y)

    x_plot = []
    y_plot = []
    z_plot = []

    x_coordinate = nn.Parameter(torch.tensor([3.0]))
    y_coordinate = nn.Parameter(torch.tensor([3.0]))

    optimizer = optim.Adam((x_coordinate,y_coordinate),lr=lr)

    plt.ion()
    for _ in range(1000):
        optimizer.zero_grad()
        loss = gradient_descent_ascent(x_coordinate,y_coordinate,f)
        loss.backward()
        optimizer.step()

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

    plt.show()