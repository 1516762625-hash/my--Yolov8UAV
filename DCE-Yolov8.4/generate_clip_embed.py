"""
运行此脚本生成 VisDrone 类别的 CLIP 文本嵌入
Usage: python generate_clip_embed.py
"""
import torch

def generate_visdrone_clip_embeddings():
    try:
        import clip
    except ImportError: 
        print("❌ 请先安装 CLIP:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        return
    
    # VisDrone 的 10 个类别
    classes = [
        "pedestrian",      # 行人
        "people",          # 人群
        "bicycle",         # 自行车
        "car",             # 汽车
        "van",             # 面包车
        "truck",           # 卡车
        "tricycle",        # 三轮车
        "awning-tricycle", # 遮阳三轮车
        "bus",             # 公交车
        "motor"            # 摩托车
    ]
    
    # 多种 prompt 模板（增强鲁棒性）
    templates = [
        "a photo of a {}",
        "a photo of a {} in the street",
        "an aerial view of a {}",
        "a drone photo of a {}",
        "a {} captured from above",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📍 使用设备: {device}")
    
    # 加载 CLIP
    print("🔄 加载 CLIP 模型...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # 提取特征（多模板平均）
    print("🔄 提取文本特征...")
    all_features = []
    
    with torch.no_grad():
        for cls_name in classes:
            cls_features = []
            for template in templates:
                text = template.format(cls_name)
                text_input = clip.tokenize([text]).to(device)
                feature = model.encode_text(text_input)
                cls_features.append(feature)
            
            # 对同一类别的多个模板取平均
            cls_feature = torch.stack(cls_features).mean(dim=0)
            all_features.append(cls_feature)
    
    # 合并并归一化
    text_features = torch.cat(all_features, dim=0)  # [10, 512]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 保存
    save_path = "visdrone_clip_embed.pt"
    torch.save(text_features.float().cpu(), save_path)
    
    print(f"✅ 保存成功！")
    print(f"   路径: {save_path}")
    print(f"   形状: {text_features.shape}")
    print(f"   类别:  {classes}")
    
    # 验证类别间相似度
    print("\n📊 类别间相似度矩阵:")
    sim_matrix = text_features @ text_features.t()
    for i, cls_i in enumerate(classes):
        sim_str = " ".join([f"{sim_matrix[i, j].item():.2f}" for j in range(len(classes))])
        print(f"   {cls_i: 16s}: {sim_str}")


if __name__ == "__main__":
    generate_visdrone_clip_embeddings()