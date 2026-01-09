#!/usr/bin/env python3
"""Download required models for TTC Security Attacks."""
import argparse
import sys

def download_prm_models():
    """Download PRM models from HuggingFace."""
    print("Downloading PRM models...")
    models = [
        "peiyi9979/math-shepherd-mistral-7b-prm",
        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
    ]
    
    try:
        from huggingface_hub import snapshot_download
        for model in models:
            print(f"  Downloading {model}...")
            # snapshot_download(model)  # Uncomment to actually download
            print(f"  {model} ready")
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")

def main():
    parser = argparse.ArgumentParser(description='Download models')
    parser.add_argument('--prm', action='store_true', help='Download PRM models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    args = parser.parse_args()
    
    if args.all or args.prm:
        download_prm_models()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
