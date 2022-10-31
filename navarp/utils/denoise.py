import os
from os.path import dirname
import numpy as np
import torch
from navarp.extras import srcnn



def denoise_data(data):
    outputs = []
    path_navarp = dirname(dirname(__file__))
    path_extras = os.path.join(path_navarp, 'extras')
    path_navarp_denoise_model = os.path.join(path_extras, 'weights.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = srcnn.SrCNN().to(device)
    model.load_state_dict(torch.load(path_navarp_denoise_model, map_location=device))
    model.eval()
    print("data shape: ", data.shape)
    N = data.shape[0]
    for i in range(N):
        print("denoising slice: ", i)
        noise = torch.tensor(data).to(device)
        out = model(noise.unsqueeze(1).unsqueeze(1))
        outputs.append(out.squeeze(1).detach().cpu().numpy())
        denoised_spectra = np.array(outputs)
        
        print("output shape: ", denoised_spectra[0,:,0,:].shape)

        return denoised_spectra[0,:,0,:]