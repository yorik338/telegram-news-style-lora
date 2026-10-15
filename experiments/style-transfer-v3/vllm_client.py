"""Exact v2 training prompt over vLLM; raw response only."""
import argparse,json,os
from pathlib import Path
from urllib.request import Request,urlopen
from common import messages

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--style',type=int,choices=[1,2],required=True)
    ap.add_argument('--base-url',default='http://127.0.0.1:8000/v1');a=ap.parse_args()
    payload=dict(model='telegram-style-v3',messages=messages(a.source.read_text(encoding='utf-8'),a.style),
        temperature=0,max_tokens=384,repetition_penalty=1,seed=42)
    request=Request(a.base_url.rstrip('/')+'/chat/completions',json.dumps(payload).encode('utf-8'),
        {'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
    with urlopen(request,timeout=300) as response:result=json.load(response)
    if result['choices'][0]['finish_reason']=='length':raise RuntimeError('Incomplete model output')
    print(result['choices'][0]['message']['content'])
if __name__=='__main__':main()
