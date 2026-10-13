"""Same source-grounded prompt over a vLLM OpenAI-compatible endpoint."""
import argparse,json,os
from pathlib import Path
from urllib.request import Request,urlopen
from pipeline import messages
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--style',type=int,choices=[1,2],required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');a=p.parse_args()
    body=dict(model='telegram-style',messages=messages(a.source.read_text(encoding='utf-8'),a.style),temperature=.6,top_p=.95,top_k=20,
        repetition_penalty=1,max_tokens=512,seed=42)
    req=Request(a.base_url.rstrip('/')+'/chat/completions',json.dumps(body).encode('utf-8'),
        {'Content-Type':'application/json','Authorization':'Bearer '+os.environ['VLLM_API_KEY']})
    with urlopen(req,timeout=300) as response:r=json.load(response)
    print(r['choices'][0]['message']['content'])
if __name__=='__main__':main()
