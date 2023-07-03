from LTE import LTE
import torch
from functools import partial
from SearchTransformer import SearchTransformer
from mmagic.models.archs import ResidualBlockNoBN
from mmcv.cnn import build_conv_layer
from refine_ref import RefineRef

patch_lq_up = torch.Tensor(2, 3, 226, 226)
ref_down_up = torch.Tensor(2, 3, 226, 226)
ref = torch.Tensor(2, 3, 226, 226)

lte = LTE()
st = SearchTransformer()

out_1 = lte(patch_lq_up)
out_2 = lte(ref_down_up)
out_3 = lte(ref)

soft_attn, texture = st(out_1, out_2, out_3)

print(soft_attn.shape, texture.shape)

print("-------------------------------")

refine_ref = RefineRef(3, 64, 64, 20, 1.)
output = refine_ref(patch_lq_up, soft_attn, texture)

print(output.shape)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print("LTE params: ", get_n_params(lte))
print("Transformer params: ", get_n_params(st))
print("RefineRef params: ", get_n_params(refine_ref))