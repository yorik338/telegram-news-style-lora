"""Match the tokenizer's fixed assistant format before generating actual post content."""
def prompt_ids(tokenizer,messages):
    header=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
    empty=tokenizer.apply_chat_template(messages+[dict(role='assistant',content='')],tokenize=True)
    if empty[:len(header)]!=header:raise ValueError('Assistant generation header differs from full chat encoding')
    suffix=empty[len(header):]
    if tokenizer.eos_token_id not in suffix:raise ValueError('Cannot identify the empty assistant end marker')
    fixed=suffix[:suffix.index(tokenizer.eos_token_id)]
    # This prefix is entirely tokenizer-defined, independent of target text or facts.
    return header+fixed,fixed
