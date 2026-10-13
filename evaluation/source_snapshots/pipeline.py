"""One trained adapter, two source-grounded Russian news styles."""
import contextlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def messages(source,style,naive=False):
    if style not in (1,2) or not isinstance(source,str) or not source.strip():raise ValueError('Source and style 1/2 required')
    if naive:
        style_text='аналитического Telegram-канала, без эмодзи, с подписью @banki_oil' if style==1 else 'городского Telegram-канала, с эмодзи и короткими абзацами'
        return [dict(role='system',content='Ты редактор русскоязычных новостей.'),
            dict(role='user',content=f'Перепиши новость в стиле {style_text}. Сохрани факты. Верни готовый пост.\n\n'+source)]
    p=json.loads((ROOT/'prompt_template.json').read_text(encoding='utf-8'));e=p['examples'][str(style)]
    return [dict(role='system',content=p['SYSTEM']+'\nСТИЛЬ: '+p['STYLE'][str(style)]),
        dict(role='user',content='ПРИМЕР СТИЛЯ ДРУГОЙ НОВОСТИ. НЕ ИСТОЧНИК НОВОЙ НОВОСТИ.\n[EX1]:\n'+e['source']),
        dict(role='assistant',content=e['response']),
        dict(role='user',content='НОВАЯ САМОСТОЯТЕЛЬНАЯ НОВОСТЬ. Единственный её источник [S1]:\n'+source+'\nНапиши пост по [S1] на русском. Сохрани авторство заявлений и степень уверенности источника.')]

def render(obj,style):
    if 'text' in obj:return obj['text']
    if style==1:return obj['headline'].strip().rstrip('.!?')+'. '+' '.join(obj['paragraphs'])+' @banki_oil'
    return '📰 **'+obj['headline'].strip()+'**\n\n'+'\n\n'.join(obj['paragraphs'])

def load_base(path=None,revision=None):
    import torch
    from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
    lock=json.loads((ROOT/'model_lock.json').read_text(encoding='utf-8'))
    revision=revision or lock['revision']
    if revision!=lock['revision']:raise ValueError('Use the pinned training base revision')
    path=path or lock['model'];config=AutoConfig.from_pretrained(path,revision=revision)
    config.quantization_config['bnb_4bit_compute_dtype']='float16'
    model=AutoModelForCausalLM.from_pretrained(path,revision=revision,config=config,dtype=torch.float16,
        device_map={'':0},attn_implementation='sdpa').eval()
    return model,AutoTokenizer.from_pretrained(path,revision=revision)

class Rewriter:
    def __init__(self,base_path=None):
        from peft import PeftModel
        self.model,self.tokenizer=load_base(base_path)
        self.model=PeftModel.from_pretrained(self.model,ROOT/'adapter').eval()
    def generate(self,source,style,variant='solution',feedback=None):
        import torch
        if variant not in ('solution','base','naive'):raise ValueError('Unknown variant')
        msg=messages(source,style,naive=variant=='naive')
        if feedback:msg[-1]['content']+='\nРедакторская проверка предыдущего ответа: '+feedback+'\nСформулируй новый пост, опираясь только на S1.'
        inputs=self.tokenizer.apply_chat_template(msg,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
        n=inputs['input_ids'].shape[1]
        if n+512>4096:raise ValueError('No evidence truncation; input+output must fit 4096')
        torch.manual_seed(42);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        ctx=self.model.disable_adapter() if variant!='solution' else contextlib.nullcontext()
        with ctx,torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=512,do_sample=True,temperature=.6,top_p=.95,top_k=20,
                repetition_penalty=1,pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize();tokens=out[0,n:];text=self.tokenizer.decode(tokens,skip_special_tokens=True)
        return dict(response=text,variant=variant,style=style,source=source,messages=msg,prompt_tokens=n,output_tokens=len(tokens),
            truncated=len(tokens)==512,seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),technical_error=None)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--style',type=int,choices=[1,2],required=True);a.add_argument('--base-path');args=a.parse_args()
    print(Rewriter(args.base_path).generate(args.source.read_text(encoding='utf-8'),args.style)['response'])
