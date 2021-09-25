import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import seaborn as sns

#https://arxiv.org/pdf/2103.12685.pdf



class network(nn.Module):
    def __init__(self,input_size=1,output_size=1,hidden_size=100):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,output_size))
    
    def forward(self,x):
        return self.layers(x)


def data(samples_per_mode,mean_1=-1,std_1=0.1,mean_2=0,std_2=0.1,mean_3=1,std_3=0.1):    
    #3 modse so samples_per_mode* 3 = amount of samples

    samples = torch.Tensor([])
    
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_1 + mean_1),dim=0)
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_2 + mean_2),dim=0)
    samples = torch.cat((samples,torch.randn(samples_per_mode,1)*std_3 + mean_3),dim=0)
    return samples 


if __name__ == "__main__":
    
    #hyperparam
    lr_gen = 0.001
    lr_disc = 0.001
    generator_latent_size = 10
    samples_per_mode = 1000
    amount_samples = samples_per_mode*3
    epochs = 10000
    k_steps = 5 #discriminator updates
    #init
    generator = network(input_size=generator_latent_size)
    discriminator = network()


    optimizer_generator = optim.Adam(generator.parameters(),lr=lr_gen)
    optimizer_disciminator = optim.Adam(discriminator.parameters(),lr=lr_disc)
    
    fig, axis = plt.subplots(3,1)
    loss_desc_plot = []
    loss_gen_plot = []
    plt.ion()
    for _ in range(epochs):
        cumulative_desc_loss = 0
        for _ in range(k_steps): #ascent 
            optimizer_disciminator.zero_grad()
            samples = data(samples_per_mode)
            loss_disc = -(torch.sum(torch.log(nn.Sigmoid()(discriminator(samples))) + torch.log((1-nn.Sigmoid()(discriminator(generator(torch.randn(amount_samples,generator_latent_size)))))))/amount_samples)
            loss_disc.backward()
            cumulative_desc_loss += loss_disc.detach().item()
            optimizer_disciminator.step()
        loss_desc_plot.append(cumulative_desc_loss/k_steps)
       
        optimizer_generator.zero_grad()
        loss_gen = torch.sum(torch.log((1-nn.Sigmoid()(discriminator(generator(torch.randn(amount_samples,generator_latent_size)))))))/amount_samples
        loss_gen.backward()
        loss_gen_plot.append(loss_gen.detach().item())
        optimizer_generator.step()

        #cla
        for ax in axis:
            ax.cla()
        #plot
        sns.kdeplot(generator(torch.randn(amount_samples,generator_latent_size)).view(-1).detach(),ax=axis[0])
        sns.kdeplot(data(amount_samples).view(-1),ax=axis[1])    
        axis[2].plot(torch.arange(len(loss_desc_plot)),loss_desc_plot,label="discriminator loss")
        axis[2].plot(torch.arange(len(loss_gen_plot)),loss_gen_plot,label="generator loss")
        axis[2].legend()
        plt.pause(0.001)
    





