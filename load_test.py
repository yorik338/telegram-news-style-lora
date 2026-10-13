"""15 distinct near-4K requests; records errors and latency, never invents a run."""
import argparse,concurrent.futures,json,os,statistics,threading,time
from urllib.request import Request,urlopen
from pipeline import ROOT

def main():
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');p.add_argument('--concurrency',type=int,default=15)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    t=AutoTokenizer.from_pretrained(ROOT/'adapter')
    prepared=[]
    for i in range(a.concurrency):
        # Distinct prefixes prevent a shared article prefix from hiding the worst KV demand.
        ids=t.encode(f'Уникальный запрос {i}. Продолжай до лимита. ',add_special_tokens=False)
        filler=t.encode(' Нейтральный текст новости для проверки нагрузки.',add_special_tokens=False)
        while len(ids)<3600:ids.extend(filler)
        ids=ids[:3560]
        msg=[dict(role='user',content=t.decode(ids))]
        count=len(t.apply_chat_template(msg,add_generation_prompt=True,tokenize=True))
        while count>3584:
            ids=ids[:-4];msg=[dict(role='user',content=t.decode(ids))];count=len(t.apply_chat_template(msg,add_generation_prompt=True,tokenize=True))
        assert 3500<=count<=3584
        prepared.append((msg,count))
    if a.prepare_only:
        print(json.dumps(dict(concurrency=a.concurrency,prompt_token_counts=[n for _,n in prepared],output_tokens=512,
            total_context_limit=4096,http_requests_sent=0),indent=2));return
    key=os.environ['VLLM_API_KEY'];records=[]
    for batch in range(a.rounds):
        barrier=threading.Barrier(a.concurrency)
        def call(i):
            msg,n=prepared[i];body=dict(model='telegram-style',messages=msg,temperature=0,max_tokens=512,ignore_eos=True,seed=batch*100+i)
            req=Request(a.base_url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
            barrier.wait();start=time.perf_counter()
            try:
                with urlopen(req,timeout=600) as r:reply=json.load(r)
                usage=reply.get('usage',{});actual=usage.get('prompt_tokens',0)
                assert actual+usage.get('completion_tokens',0)<=4096
                assert usage.get('completion_tokens')==512,'Must exercise worst-case completion budget'
                return dict(batch=batch,id=i,ok=True,seconds=time.perf_counter()-start,local_prompt_tokens=n,usage=usage)
            except Exception as e:return dict(batch=batch,id=i,ok=False,seconds=time.perf_counter()-start,error=str(e))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:records.extend(pool.map(call,range(a.concurrency)))
    times=[r['seconds'] for r in records if r['ok']]
    result=dict(requests=len(records),ok=sum(r['ok'] for r in records),errors=sum(not r['ok'] for r in records),
        median_seconds=statistics.median(times) if times else None,records=records)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['errors']:raise SystemExit(1)
if __name__=='__main__':main()
