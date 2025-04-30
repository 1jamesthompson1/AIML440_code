The real goal of this project is working on making some novel contribution to the space of AI in the subfield of Reinforcement Learning within the context of Continuous control.

Here are some ideas on what I could look into and investigate:

## Ensemble of actors

Related works:
https://arxiv.org/pdf/1902.05551
https://arxiv.org/pdf/2209.14488


Currently there is work like REDQ, DroQ and such that use ensemble in the critic networks. However it seems like using ensemble policies could also have benfits. Potentialyl even doing away with the critics.

If I stick with the Actor critic framework then I have choices for how to train the polices and how to select an action.

This is remininst of the idea of many minds that is from psychology

I could have a single critic. Then this critic will give me the expected value of each action offered by the polices. It will select the action that gives the highest expected reward. I need to make sure that the polcies stay different which could be wiht something like KL divergence.
The polices can be updated towards a target made by the critic. However this will make all of thep olices tend to be the same so some diveregence loss will need to be added where they must be as close to the target while as far away from each other as possible. Make a sort of high dimensional circle around the optimal policy. This will be more like a subset version of strict Q learning. 


Multihead actors or completely seperate actors
The diffenet actos can be thought of as experts in their own special domain. 

Alogrithm

1. Take action based on best policy. This best policy can either be from the action that has the highest expected reward according to Q function see note 1, or the average of the policies (maybe element wise mean or centre of mass).
2. Update the Q function using the previous used policy
3. Update the policies towards the Q function that has a entropy maximisation and a push to be different to all the other polices.


Note 1
If we use the highest expected reward then all the policy functions do is provide us with a subset of possible actions to look at while doing more tradtional Q-learning. 

## Attention mechanism

Something like in transformers that mean the next toke npredicted can more heavily weightrelevantprevioous words. The next action taken can be weighted by more relvant current or past observation values. I.e let it deconstruct the input alittle bit using the attention mechamissm. There seems to things like that here: https://arxiv.org/pdf/2503.03660 with a trasnformer based SAC.

## Generalisation

Currently the conintunous control task is being trained on a single envrionment and then ahs to be complteetly trained from scratch for a new envrionment.

There is metalearning which is about tasks: https://github.com/Farama-Foundation/Metaworld. However it is centered around tasks where the agent stays the same., I.e learning ap olicy that can quickly learn new task polices well.

It would be good to take this one step lower and learn something like a pretrained model that can then transfer to different robots. For example do well on all of the environments.
- Need some way of having an intrinsic understanding of angles, gravity, such. As the inputs will be ocmpltely different and the outputs as well will be completely different.
- How to force it to stay really general
