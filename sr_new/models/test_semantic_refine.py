from SemanticRefine import EnhanceNetwork, GrowthAttentionResidualBlock
import torch

# x = torch.Tensor(1, 64, 64, 64)
# models = GrowthAttentionResidualBlock(mid_channels=64, growth_channels=32)
# x1 = models.lrelu(models.attn1(models.conv1(x)))
# x2 = models.lrelu(models.attn2(models.conv2(torch.cat((x, x1), dim=1))))
# output = models.attn3(models.conv3(torch.cat((x, x1, x2), 1)))
# print(output.shape)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

x = torch.Tensor(1, 3, 64, 64)
sem = torch.Tensor(1, 1, 64, 64)
model = EnhanceNetwork(64, 20)

output = model.forward(x, sem)

print(get_n_params(model))

# print(output.shape)