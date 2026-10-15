"""Inference with the v2 adapter and its exact training prompt; no editorial edits."""
import argparse
from pathlib import Path
from common import ROOT,load_base,messages,generate

def main():
    from peft import PeftModel
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--style',type=int,choices=[1,2],required=True)
    ap.add_argument('--base-path');ap.add_argument('--base-only',action='store_true');a=ap.parse_args()
    model,tok=load_base(a.base_path)
    if not a.base_only:model=PeftModel.from_pretrained(model,ROOT/'adapter',autocast_adapter_dtype=False).eval()
    result=generate(model,tok,messages(a.source.read_text(encoding='utf-8'),a.style))
    if result['truncated']:raise RuntimeError('Incomplete response; never publish silently')
    print(result['response'])
if __name__=='__main__':main()
