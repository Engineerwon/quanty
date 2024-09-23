import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class LMClass(BaseLM): #허깅페이스에서 모델 불러와서 써먹기 위한 코드라는데?
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        if not ("bert" in args.net.lower()):
            config = AutoConfig.from_pretrained(
                args.model, attn_implementation=args.attn_implementation
            )
        else:
            config = AutoConfig.from_pretrained(args.model, num_labels=2)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        """bert를 causalLM으로 불러올 수 없다. AutoModelForSequenceClassification을 써야된다. 인자는 똑같이 써도 될듯"""
        if "bert" in args.net.lower(): #일단은 이진분류만 하자.
            self.model = AutoModelForSequenceClassification.from_pretrained(args.model,config=config, device_map='cpu',torch_dtype=torch.float16)
        else:    
            self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property ##이 밑부터는 각각 메서드들 정의. 
    def eot_token(self) -> str: #end of text 토큰 리턴
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self): #end of text 토큰의 id를 리턴
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self): #모델이 허용하는 최대 문장 길이 반환.
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self): #한번에 생성할 수 있는 최대 토큰 수 반환
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self): 
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
