import torch
from src.models.diffusion_net import DiffusionPolicy
def test_padding_fix():
    # 16 Horizon 데이터가 Padding 로직을 통해 에러 없이 통과되는지 확인
    m = DiffusionPolicy(14, 14)
    o = m(torch.randn(2, 16, 14), torch.tensor([1,1]), torch.randn(2, 14))
    assert o.shape == (2, 16, 14)