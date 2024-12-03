from fit import fit
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import math
import yaml

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
    
msamples = [20, 50, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000]

losses = []
nepochs = []

for msample in msamples:
    config['max_samples'] = msample
    config['nepoch'] = (math.ceil(math.log(msample, 2)) + 1) * config['schedule_interval'] + 1
    nepochs.append(config['nepoch'])
    # breakpoint()
    rc, gt = fit(config)
    with torch.no_grad():
        loss = nn.MSELoss()(rc, gt)
    losses.append(loss)
    
    log = open('log/log_exp.txt', 'w')
    log.write(str(msamples[:len(losses)]) + '\n')
    log.write(str(nepochs) + '\n')
    log.write(str(losses) + '\n')
    log.close()

    plt.plot(msamples[:len(losses)], losses)
    plt.xlabel('Number of Gaussians')
    plt.ylabel('MSE Loss')
    plt.savefig('log/loss_exp.png')
    

