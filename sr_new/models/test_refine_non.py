from refine_non_ref import RefineNonRef
import torch

inp = torch.Tensor(1, 3, 64, 64)

model = RefineNonRef(3, 64, 15, 1)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(model))

output = model(inp)

print(output.shape)