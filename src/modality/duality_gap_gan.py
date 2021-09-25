import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import chain 
#https://arxiv.org/pdf/2103.12685.pdf

class generator(nn.Module):
    def __init__(self,noise_size=10,output_size=1,hidden_size=100):
        super().__init__()
        self.noise_size = noise_size
        self.layers = nn.Sequential(nn.Linear(noise_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size))
    
    def forward(self,x):
        return self.layers(x)

class discriminator(nn.Module):
    def __init__(self,input_size=1,output_size=1,hidden_size=100):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size))
    
    def forward(self,x):
        return nn.Sigmoid()(self.layers(x))

def data(samples_per_mode,mean_1=-1,std_1=0.1,mean_2=0,std_2=0.1,mean_3=1,std_3=0.1):    
    #3 modse so samples_per_mode* 3 = amount of samples

    samples = torch.Tensor([])
    
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_1 + mean_1),dim=0)
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_2 + mean_2),dim=0)
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_3 + mean_3),dim=0)
    return samples 

def loss_generator(samples,disc,gen):
    batch_size = samples.size(0)
    rand_vec = torch.randn(batch_size,gen.noise_size)
   
    return torch.sum(torch.log(1 - disc(gen(rand_vec))))/batch_size

def loss_discriminator(samples,disc,gen):

    batch_size = samples.size(0)
    rand_vec = torch.randn(batch_size,gen.noise_size)
    return -(torch.sum(torch.log(disc(samples)) + torch.log(1 - disc(gen(rand_vec))))/batch_size) #- is as this is gradient ascent
     

def duality_loss(samples,inner_disc,disc,inner_gen,gen):
    return loss_generator(samples,inner_disc,gen) + loss_discriminator(samples,disc,inner_gen)


if __name__ == "__main__":
    #hyper param
    lr_inner = 0.001
    lr_outer = 0.01
    k_steps = 20 #how many innersteps
    samples_per_mode = 1000
    amount_samples = samples_per_mode*3
    epochs = 10000
    #models and optimizers
    disc = discriminator()
    gen = generator()

    duality_optim = optim.Adam(chain(disc.parameters(),gen.parameters()),lr=lr_outer)

    #train loop
    fig, axis = plt.subplots(3,1) 
    
    duality_loss_plot = []
    plt.ion()
    for i in range(epochs):
        #OPTIMIZING INNER MODELS
        inner_gen = generator()
        inner_disc = discriminator()
        #load weights 
        inner_gen.load_state_dict(gen.state_dict())
        inner_disc.load_state_dict(disc.state_dict())
        #init inner optimizer
        optim_inner_gen = optim.Adam(inner_gen.parameters(),lr=lr_inner)
        optim_inner_disc=  optim.Adam(inner_disc.parameters(),lr=lr_inner)
        for i in range(k_steps):
            samples = data(samples_per_mode)
            optim_inner_disc.zero_grad()
            disc_loss = loss_discriminator(samples,inner_disc,gen)
            disc_loss.backward()
            optim_inner_disc.step()

            optim_inner_gen.zero_grad()
            gen_loss = loss_generator(samples,disc,inner_gen)
            gen_loss.backward()
            optim_inner_gen.step()

        #OPTIMIZING DUALITY GAP OBJECTIVE
        duality_optim.zero_grad()
        d_loss = duality_loss(samples,inner_disc,disc,inner_gen,gen)
        d_loss.backward()
        duality_optim.step()
        duality_loss_plot.append(d_loss.detach().item())

        axis[0].cla()
        axis[1].cla()
        axis[2].cla()
        sns.kdeplot(gen(torch.randn(amount_samples,gen.noise_size)).view(-1).detach(),ax=axis[0])
        sns.kdeplot(data(amount_samples).view(-1),ax=axis[1])    
        axis[2].plot(torch.arange(len(duality_loss_plot)),duality_loss_plot,label="duality loss")
        axis[2].legend()

        plt.pause(0.1)