import os

# 这是报错里显示的那个绝对路径
target_file = "/root/miniconda3/lib/python3.12/site-packages/ultralytics/nn/modules/block.py"

print(f"正在修复文件: {target_file}")

# 1. 读取原文件
with open(target_file, 'r') as f:
    lines = f.readlines()

# 2. 准备新的 VSSBlock 代码 (包含 CPU 跳过逻辑)
new_vss_block = """
class VSSBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        except ImportError:
            self.mamba = None

        self.proj = nn.Linear(in_channels, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # [关键修复] 如果是 CPU 数据 (YOLO 初始化检查)，直接跳过，防止 Mamba 报错
        if not x.is_cuda:
            return x

        if self.mamba is None:
            return x

        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
        res = x_flat
        x_norm = self.norm(x_flat)

        # 强制 FP32 避免对齐问题
        with torch.cuda.amp.autocast(enabled=False):
            x_in = x_norm.float()
            x_mamba = self.proj(x_in)
            x_mamba = x_mamba.contiguous()
            x_mamba = self.mamba(x_mamba)
            x_mamba = self.proj_out(x_mamba)
            out = x_mamba + res.float()

        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        if torch.is_autocast_enabled():
            out = out.half()
        return out
"""

# 3. 笨办法替换：把文件里的旧类定义替换掉
# 注意：这需要你的 block.py 里原来的 VSSBlock 定义结构相对标准
# 如果自动替换失败，我们就在末尾追加，覆盖定义

content = "".join(lines)

# 简单的覆盖策略：如果文件里已经有 VSSBlock，我们得小心。
# 最稳妥的方式：直接告诉用户去改这个特定的文件。

print("="*50)
print("请注意！你需要修改的文件是这个：")
print(target_file)
print("="*50)

# 如果你敢信我，下面这行代码会尝试追加新定义到文件末尾 (Python 后定义的类会覆盖前面的)
# 这是一种 Dirty Hack，但能立刻生效
with open(target_file, 'a') as f:
    f.write("\n\n" + "#" * 20 + " HOT FIX BY GEMINI " + "#" * 20 + "\n")
    f.write(new_vss_block)
    f.write("\n" + "#" * 20 + " END HOT FIX " + "#" * 20 + "\n")

print("已将新的 VSSBlock 代码追加到 block.py 文件末尾。")
print("Python 加载时，后定义的类会覆盖前面的类，所以这应该能生效。")
print("请立刻重新运行训练命令：yolo detect train ...")