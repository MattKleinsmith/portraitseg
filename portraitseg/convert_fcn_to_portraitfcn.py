import torch
from torchfcn.models import FCN8s

from portraitseg.portraitfcn import PortraitFCN


portraitfcn = PortraitFCN()

fcn8s = FCN8s()
path_to_weights = "portraitseg/fcn8s_from_caffe.pth"
fcn8s.load_state_dict(torch.load(path_to_weights))

fcn8s_params = fcn8s.state_dict()
chosen_parameters = ['score_fr.weight',
                     'score_fr.bias',
                     'score_pool3.weight',
                     'score_pool3.bias',
                     'score_pool4.weight',
                     'score_pool4.bias',
                     'upscore2.weight',
                     'upscore8.weight',
                     'upscore_pool4.weight']

N = 15 # "person" class ID in PASCAL VOC is 15
for namep, portraitfcn_param in portraitfcn.named_parameters():
    print(namep)
    if namep in chosen_parameters:
        # Extract only the background channel (0) and person channel (15)
        if "upscore" in namep:
            # The upscore layers went from 21 channels to 21
            portraitfcn_param.data = fcn8s_params[namep][0:N+1:N, 0:N+1:N]
        else:
            # The score layers went from various channels to 21 channels
            portraitfcn_param.data = fcn8s_params[namep][0:N+1:N]
    else:
        # Copy the other parameters unmodified.
        portraitfcn_param.data = fcn8s_params[namep]

torch.save(portraitfcn.state_dict(), "portraitseg/portraitfcn_untrained.pth")
