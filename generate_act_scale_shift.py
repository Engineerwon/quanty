import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
try:
    from llava.model import *   # required for llava
except ImportError:
    print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# import pdb



def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        inputs = dataloader[i][0].unsqueeze(0).to(device)  # input_ids에 배치 차원 추가
        attention_mask = dataloader[i][1].unsqueeze(0).to(device)  # attention_mask에 배치 차원 추가
        token_type_ids = dataloader[i][2].unsqueeze(0).to(device) if dataloader[i][2] is not None else None  # token_type_ids에 배치 차원 추가

        if token_type_ids is not None:
            model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            model(input_ids=inputs, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        inputs = dataloader[i][0].unsqueeze(0).to(device)  # input_ids에 배치 차원 추가
        attention_mask = dataloader[i][1].unsqueeze(0).to(device)  # attention_mask에 배치 차원 추가
        token_type_ids = dataloader[i][2].unsqueeze(0).to(device) if dataloader[i][2] is not None else None  # token_type_ids에 배치 차원 추가

        if token_type_ids is not None:
            model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            model(input_ids=inputs, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    return act_shifts




# def build_model_and_tokenizer(model_name):
#     kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
#     return model, tokenizer

def build_model_and_tokenizer(model_name):
    # GPU가 있을 경우 GPU로, 그렇지 않으면 CPU로 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # BERT 모델을 불러올 때는 SequenceClassification으로 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 모델을 선택한 디바이스로 이동 (GPU 또는 CPU)
    model.to(device)
    
    return model, tokenizer
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/cpfs01/user/chenmengzhao/llama_quantization/llama-hf/llama-7b', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="sst2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataloader, _ = get_loaders(
    args.calib_dataset,
    nsamples=args.num_samples,
    seed=args.seed,
    model=args.model,
    seqlen=args.seq_len,
    )
    
    args.net = args.model.split('/')[-1]
    act_scales = get_act_scales(model, dataloader,args.num_samples)
    save_path = os.path.join(args.scales_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(model, dataloader,args.num_samples)
    save_path = os.path.join(args.shifts_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)


if __name__ == '__main__':
    main()
