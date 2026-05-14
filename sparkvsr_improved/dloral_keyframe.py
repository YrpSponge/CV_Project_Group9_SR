#!/usr/bin/env python3
"""CLI wrapper for DLoRA single-image enhancement.
Usage:
    DLORA_HOME=/path/to/DLoRA python dloral_keyframe.py         --input_image <path> --output_dir <dir>         --pretrained_path <model.pkl> --sd_model_path <sd2.1_dir>         [--sidechannel_ckpt <sc.pt>]
"""
import os, sys, argparse

DLORA_HOME = os.environ.get('DLORA_HOME', os.path.dirname(os.path.abspath(__file__)))
os.chdir(DLORA_HOME)
sys.path.insert(0, DLORA_HOME)

import numpy as np
from PIL import Image
from src.inference_wrapper import DLoRALInferenceWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to DLoRA RAFT checkpoint (e.g. model_52001.pkl)')
    parser.add_argument('--sd_model_path', type=str, required=True,
                        help='Path to stable-diffusion-2-1-base directory')
    parser.add_argument('--sidechannel_ckpt', type=str, default=None)
    parser.add_argument('--process_size', type=int, default=512)
    args = parser.parse_args()

    img = Image.open(args.input_image).convert('RGB')
    img_np = np.array(img)

    model = DLoRALInferenceWrapper(
        pretrained_path=args.pretrained_path,
        pretrained_model_path=args.sd_model_path,
        pretrained_model_name_or_path=args.sd_model_path,
        pretrained_model_path_csd=args.sd_model_path,
        flow_estimator='raft',
        sidechannel_ckpt=args.sidechannel_ckpt,
        process_size=args.process_size,
        mixed_precision='fp32',
    )

    outputs = model([img_np, img_np])
    result = Image.fromarray(outputs[0])

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'input_frame.png')
    result.save(out_path)
    print(f'DLoRA output saved to {out_path}')

if __name__ == '__main__':
    main()
