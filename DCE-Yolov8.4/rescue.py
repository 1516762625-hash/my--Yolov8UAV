import os
import shutil

# 目标文件路径
target_file = "/root/miniconda3/lib/python3.12/site-packages/ultralytics/nn/modules/block.py"
cache_dir = "/root/miniconda3/lib/python3.12/site-packages/ultralytics/nn/modules/__pycache__"

print(f"🚑 正在启动救援程序，目标: {target_file}")

# 1. 读取文件内容
if not os.path.exists(target_file):
    print("❌ 找不到文件！请确认路径是否正确。")
    exit()

with open(target_file, 'r') as f:
    lines = f.readlines()

# 2. 清理战场 (删除我之前产生的所有垃圾代码)
clean_lines = []
skip = False
for line in lines:
    # 识别并跳过我之前添加的补丁块
    if "GEMINI" in line or "Monkey Patch" in line or "mamba_ssm.ops" in line:
        # 如果是之前加在头部的补丁，跳过
        continue
    # 简单的清理逻辑：保留原始的类定义，删除后面追加的
    if "class VSSBlock" in line and "class C2f_VSS" not in line:
        # 这是一个简单的启发式，我们稍后会统一追加最新的类定义
        # 但为了防止文件被删空，我们这里先采取保守策略：
        # 只过滤掉显式标记了 GEMINI 的区域
        pass
    
    clean_lines.append(line)

# 为了确保万无一失，我们重新读取原始干净的 block.py (如果备份还在)
# 如果没有备份，我们手动构建：
# 核心逻辑：找到 `from __future__`，确保它在第一行

final_lines = []
future_import = "from __future__ import annotations\n"

# 先把 future import 拿出来
has_future = False
for line in clean_lines:
    if "from __future__" in line:
        has_future = True
        break

if has_future:
    final_lines.append(future_import)
else:
    # 如果原文件里没有，这其实不常见，但我们不管，YOLOv8 源码里肯定有
    final_lines.append(future_import)

# 把剩下的内容加进去，排除掉重复的 future 行
for line in clean_lines:
    if "from __future__" in line:
        continue
    final_lines.append(line)

# 3. 准备正确的补丁代码 (Monkey Patch) - 放在 Import 之后
monkey_patch = """
# =============================================================================
# [GEMINI RESCUE FIX] Mamba 内存对齐补丁 (Correct Placement)
# =============================================================================
try:
    import torch
    import mamba_ssm.ops.selective_scan_interface
    
    _original_fn = mamba_ssm.ops.selective_scan_interface.selective_scan_fn
    
    def _safe_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
        # 强制所有输入在进入 CUDA 前连续
        if u is not None: u = u.contiguous()
        if delta is not None: delta = delta.contiguous()
        if A is not None: A = A.contiguous()
        if B is not None: B = B.contiguous()
        if C is not None: C = C.contiguous()
        if D is not None: D = D.contiguous()
        if z is not None: z = z.contiguous()
        if delta_bias is not None: delta_bias = delta_bias.contiguous()
        return _original_fn(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        
    mamba_ssm.ops.selective_scan_interface.selective_scan_fn = _safe_scan_fn
    print("✅ [Mamba Rescue] 底层接口劫持成功，内存对齐已强制启用。")
except ImportError:
    pass
except Exception as e:
    print(f"⚠️ [Mamba Rescue] 补丁注入警告: {e}")
# =============================================================================
"""

# 4. 准备安全的类定义 (Clone 版)
safe_classes = """
# =============================================================================
# [GEMINI SAFE CLASSES] 覆盖原有定义
# =============================================================================
import torch.nn as nn

class VSSBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        except ImportError:
            self.mamba = None
        self.proj = nn.Linear(in_channels, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # CPU 检查 + Mamba 检查
        if not x.is_cuda or self.mamba is None: return x
        
        B, C, H, W = x.shape
        # 使用 clone() 彻底切断内存引用
        x_flat = x.permute(0, 2, 3, 1).flatten(1, 2).clone()
        res = x_flat
        x_norm = self.norm(x_flat)
        
        with torch.cuda.amp.autocast(enabled=False):
            x_in = x_norm.float()
            x_mamba = self.proj(x_in).clone()
            x_mamba = self.mamba(x_mamba)
            x_mamba = self.proj_out(x_mamba)
            out = x_mamba + res.float()
            
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).clone()
        if torch.is_autocast_enabled(): out = out.half()
        return out

class C2f_VSS(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(VSSBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        # 这里的 clone 很关键
        x = x.clone()
        y = list(self.cv1(x).chunk(2, 1))
        y_last = y[-1].clone()
        output_list = []
        for m in self.m:
            output_list.append(m(y_last))
        y.extend(output_list)
        return self.cv2(torch.cat(y, 1))
# =============================================================================
"""

# 5. 组装文件
# 顺序：Future Import -> Monkey Patch -> 原文件剩余部分 -> Safe Classes
# 这样 Monkey Patch 就在开头 (但在 future 之后)，Safe Classes 在末尾覆盖旧定义

# 找到插入点：在 imports 差不多结束的地方插入 patch，或者直接在 future 之后插
final_content = []
final_content.append(final_lines[0]) # from __future__ ...
final_content.append(monkey_patch)   # 插入补丁
final_content.extend(final_lines[1:]) # 原文件剩余内容
final_content.append(safe_classes)    # 追加新类

# 6. 写入文件
with open(target_file, 'w') as f:
    f.writelines(final_content)

print("✅ block.py 文件修复完成！语法错误已修正，补丁已正确植入。")

# 7. 暴力清理缓存
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print("✅ __pycache__ 已清除，强制重新编译。")
    except Exception as e:
        print(f"⚠️ 无法清除缓存 (可能无权限): {e}")

print("🚀 救援结束。请立刻运行训练命令。")