from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading

tokenizer = AutoTokenizer.from_pretrained("jihopark/colloquial")
model = AutoModelWithLMHead.from_pretrained("jihopark/colloquial", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def run_model(prompt, num, length):
    try:
        prompt = prompt.strip()
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        # model = models[model_name]
        sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=True,
                                        max_length=length,
                                        min_length=length,
                                        top_k=40,
                                        num_return_sequences=num)

        generated_texts = ""
        for i, sample_output in enumerate(sample_outputs):
            output = tokenizer.decode(sample_output.tolist()[
                                      min_length:], skip_special_tokens=True)
            generated_texts+= output

        return generated_texts

    except Exception as e:
        print(e)
        return 500

print(run_model("cake",10,50))
