#!/usr/bin/env python3
"""
日语电影转录脚本 - 专为 Codespaces 优化
使用：python transcribe.py --input 你的电影文件.mp4
"""

import whisper
import argparse
import os
import sys
import time
from whisper.utils import get_writer

def get_available_models():
    """返回可用的模型列表"""
    return ["tiny", "base", "small", "medium", "large", "large-v3"]

def estimate_time(file_size_mb, model_size):
    """粗略估计处理时间"""
    base_time_per_gb = {
        "tiny": 5,      # 分钟/GB
        "base": 10,     # 分钟/GB
        "small": 20,    # 分钟/GB
        "medium": 30,   # 分钟/GB
        "large": 45,    # 分钟/GB
        "large-v3": 50  # 分钟/GB
    }
    
    file_size_gb = file_size_mb / 1024
    est_minutes = base_time_per_gb.get(model_size, 30) * file_size_gb
    return max(5, est_minutes)  # 至少5分钟

def main():
    parser = argparse.ArgumentParser(description='使用 OpenAI Whisper 转录日语音频')
    parser.add_argument('--input', '-i', required=True, help='输入音频/视频文件路径')
    parser.add_argument('--model', '-m', default='large-v3', 
                       choices=get_available_models(),
                       help=f'模型大小，默认: large-v3 (最准确)')
    parser.add_argument('--language', '-l', default='ja', help='语言代码，默认: ja (日语)')
    parser.add_argument('--output_dir', '-o', default='./output', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误：找不到文件 '{args.input}'")
        print("请确保：")
        print("  1. 文件已上传到 Codespaces 工作区")
        print("  2. 文件名拼写正确（包括扩展名）")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 文件信息
    file_size = os.path.getsize(args.input) / (1024 * 1024)  # MB
    est_time = estimate_time(file_size, args.model)
    
    print("=" * 60)
    print("🎬 Whisper 日语转录工具")
    print("=" * 60)
    print(f"📁 输入文件: {args.input}")
    print(f"📊 文件大小: {file_size:.1f} MB")
    print(f"🤖 使用模型: {args.model}")
    print(f"🗣️  识别语言: {args.language}")
    print(f"⏳ 预计时间: {est_time:.0f} 分钟")
    print("=" * 60)
    
    # 确认继续
    response = input("是否开始转录？(y/n): ")
    if response.lower() != 'y':
        print("取消操作。")
        sys.exit(0)
    
    # 开始处理
    print(f"\n🔧 加载模型 {args.model}...")
    start_load = time.time()
    model = whisper.load_model(args.model)
    load_time = time.time() - start_load
    print(f"✅ 模型加载完成 ({load_time:.1f} 秒)")
    
    print(f"\n🚀 开始转录...")
    print("   进度条显示音频片段处理，请耐心等待！")
    
    start_transcribe = time.time()
    result = model.transcribe(
        args.input,
        language=args.language,
        verbose=True,          # 显示进度条
        fp16=True,             # GPU加速
        task="transcribe",
        initial_prompt="这是一部日语电影，包含清晰的对话。请准确转录。"
    )
    transcribe_time = time.time() - start_transcribe
    
    print(f"\n✅ 转录完成！总耗时: {transcribe_time/60:.1f} 分钟")
    
    # 准备输出文件名
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存 SRT 字幕
    srt_filename = f"{base_name}_{args.language}_{timestamp}.srt"
    srt_path = os.path.join(args.output_dir, srt_filename)
    writer = get_writer("srt", args.output_dir)
    writer(result, srt_path)
    print(f"📄 字幕文件: {srt_path}")
    
    # 保存纯文本
    txt_filename = f"{base_name}_{args.language}_{timestamp}.txt"
    txt_path = os.path.join(args.output_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"📝 文本文件: {txt_path}")
    
    # 预览结果
    print("\n" + "=" * 60)
    print("📋 预览（前500字符）:")
    print("=" * 60)
    preview = result["text"][:500]
    print(preview + ("..." if len(result["text"]) > 500 else ""))
    print("=" * 60)
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"   字符数: {len(result['text'])}")
    print(f"   处理速度: {len(result['text'])/(transcribe_time+0.1):.1f} 字符/秒")
    
    print(f"\n🎉 全部完成！文件已保存到 '{args.output_dir}' 目录。")
    print("💡 提示：右键点击文件选择 'Download' 下载到本地。")

if __name__ == "__main__":
    main()
