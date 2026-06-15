#!/usr/bin/env python3
from modelscope import snapshot_download
import os

# 确保目标文件夹存在
base_dir = '/root/autodl-tmp/models'
os.makedirs(base_dir, exist_ok=True)

print("🚀 准备从 ModelScope 高速拉取 Qwen2.5 模型...")

# 1. 下载 7B 模型
print("\n📦 正在下载 [Qwen2.5-7B-Instruct] ...")
snapshot_download(
    model_id='qwen/Qwen2.5-7B-Instruct',
    local_dir=f'{base_dir}/Qwen2.5-7B-Instruct'
)
print("✅ 7B 模型下载完成！")

# 2. 下载 3B 模型
print("\n📦 正在下载 [Qwen2.5-3B-Instruct] ...")
snapshot_download(
    model_id='qwen/Qwen2.5-3B-Instruct',
    local_dir=f'{base_dir}/Qwen2.5-3B-Instruct'
)
print("✅ 3B 模型下载完成！")

print("\n🎉 所有基座模型已就绪，可以开始炼丹了！")