# zero_competition_gan
Optimizing min-max games using duality gap objective. </br>
This code is based upon (Generative Minimization Networks: Training GANs Without Competition; arXiv:2103.12685)

Function is $$3x^{2} - y^{2} + 4xy$$. </br>
Optimizing using the duality gap objective. </br> 
![](duality_gap_objective.gif)
</br> 
Optimizing using gradient ascent-descent.</br> 
![](min_max_optimization.gif)
</br>
Objective is to learn the generator to generate samples from 3 guassians with std=1 and means= (-1,1,1) </br>
With the duality gap objective </br>
![](duality_modality.gif)
</br>
With the min-max objective. Original GAN:(Generative Adversarial Nets arXiv:1406.2661) </br>
![](gan_modality.gif)
