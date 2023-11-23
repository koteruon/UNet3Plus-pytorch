import torch
from torchvision import models
from torchviz import make_dot

from model import U3PResNetEncoder, UNet3Plus

# 使用一個例子模型，你可以替換成你的模型
encoder = U3PResNetEncoder(backbone="resnet34", pretrained=False)
transpose_final = True
fast_up = True
model = UNet3Plus(21, 64, 2, encoder, use_cgm=False, dropout=0.3, transpose_final=transpose_final, fast_up=fast_up)

# 定義一個虛擬的輸入
dummy_input = torch.randn(16, 3, 512, 512)

# 將模型的架構可視化為圖片
dot = make_dot(
    model(dummy_input)["final_pred"], params=dict(model.named_parameters()), show_attrs=True, show_saved=True
)

# 保存圖片
dot.render("model_architecture", format="png", cleanup=True)
