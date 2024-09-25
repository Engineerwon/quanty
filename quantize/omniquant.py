import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.int_bert_layer import QuantBertLayer_1, QuantBertLayer_12, QuantBertLayer_2to11
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils_
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module): #모델에서 QuantLinear타입의 레이어를 찾아서 이름과 함께 dict형태로 반환.
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module): #새로운 모듈의 추가
    levels = name.split('.') #모듈 이름을 .을 기준으로 분리하여 각 수준(level)을 리스트로 저장합니다.예를 들어, name이 "layer1.conv1"라면 levels는 ["layer1", "conv1"]가 됩니다.
    if len(levels) > 1: #분리된 이름의 레벨이 1보다 크면(즉, 모듈이 계층적으로 중첩되어 있으면), 해당 모듈을 찾아갑니다.
        mod_ = original_module
        for l_idx in range(len(levels)-1): 
            if levels[l_idx].isdigit(): #만약 현재 level이 숫자라면,
                mod_ = mod_[int(levels[l_idx])] #그것을 가져온다?
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module) # mod_라는 객체에 levels[-1]이라는 속성을 추가하고 그 속성에 added_module이라는 값을 부여.
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device #cuda
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower(): #.lower는 문자열을 소문자로 변환. 소문자로 변환한 문자열에 llama가 포함되어 있는지
        is_llama = True
        layers = model.model.layers #이건 AutoModelForCausalLM.from_pretrained(args.model, ...).layers이다.
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    elif 'bert' in args.net.lower():
        layers = model.bert.encoder.layer
        model.bert.embeddings = model.bert.embeddings.to(dev)
        DecoderLayer_1 = QuantBertLayer_1
        DecoderLayer_12= QuantBertLayer_12
        DecoderLayer_2to11 = QuantBertLayer_2to11
        layer_name_prefix = "bert.encoder.layer"
        pairs = {
            "query":"qkv",
            "attention.output.dense":"out",
            "intermediate.dense":"fc1"
        }
        pairs_for_firstblock = {
            "attention.output.dense":"out",
            "intermediate.dense":"fc1"
        }

        



    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev) #첫번째 layer를 cuda로
    if args.deactive_amp and args.epochs>0: #amp 기능을 사용한다.
        dtype = torch.float
        traincast = nullcontext
    else:
        print("amp확인\n")
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    #입력 텐서 생성
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    #nsamples가 양자화를 위해 샘플링할 입력 데이터 개수라고?
    #cache는 샘플 인덱스와 정보를 기록하는 임시 저장소? 
    cache = {"i": 0}
    masks = torch.zeros(
        (args.nsamples, 1, lm.seqlen, lm.seqlen), dtype=dtype, device=dev
    )

    # catch the first layer input, 첫번째 레이어에 입력되는 텐서를 저장.
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self,
        inp,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions):
            inp=inp.squeeze(0)

            inps[cache["i"]] = inp
            
            masks[cache["i"]] = attention_mask.squeeze(0)
            cache["i"] += 1

            

            # # 만약 llama 모델이라면 position_ids도 저장
            # if self.is_llama and "position_ids" in kwargs:
            #     cache["position_ids"] = kwargs["position_ids"]

            raise ValueError  # 예외를 발생시켜 실행을 중지

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    """여기 model에다가 입력 넣을 때가 문제가 되었음."""
    with torch.no_grad():
        for batch in dataloader: 
           

            if cache["i"] >= args.nsamples:
                break #break가 걸리면 루프를 빠져나오고, 다음 코드를 수행. nsamples만큼의 입력만...
            try:
                inputs = batch[0].unsqueeze(0).to(dev)  # input_ids에 배치 차원 추가

                attention_mask = batch[1].unsqueeze(0).to(dev)  # attention_mask에 배치 차원 추가
                token_type_ids = batch[2].unsqueeze(0).to(dev) if batch[2] is not None else None  # token_type_ids에 배치 차원 추가
                if token_type_ids is not None:
                    model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    model(input_ids=inputs, attention_mask=attention_mask)

                # model(batch[0].to(dev)) #일단 처음에 모델에다가 첫 입력을 넣음. 근데, 이때 첫번째 layer가 호출되는데, 첫번째 layer는 catcher가 씌워져있기 때문에 cathcer의 forward메서드가 수행된다.
            except ValueError: #의도적으로 catcher의 forward에서 valueError가 뜨게 함. 근데 이걸 예상하고 뜨면 pass하도록 함.
                pass
    #결과적으로, 각 배치의 첫번째 샘플만 nsamples개만큼 뽑아서 하나의 배치를 만드는 느낌? 그리고 그 결과는 inps라는 텐서임. (nsamples, seqlen, hidden size)





    # move embedding layer and first layer to cpu. 양자화할때 필요없는 레이어는 cpu로 보내서 gpu메모리 절약?
    layers[0] = layers[0].module #캐쳐 벗김.
    layers[0] = layers[0].cpu() #cpu로 옮겨줍니다.
    if "llama" in args.net.lower() or "mixtral" in args.net.lower(): #라마만 일단 볼겁니다
        model.model.embed_tokens = model.model.embed_tokens.cpu() #임베딩과 layernorm 부분도 cpu로 옮겨줍니다
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    elif "bert" in args.net.lower():
        model.bert.embeddings = model.bert.embeddings.cpu()

        """bert부분도 필요. 왜 임베딩을 아까 cuda로 보냈다가 여기선 다시 cpu로 보내는지는 이후에 알아봐야 한다."""

    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache() #gpu메모리에 있는 불필요한 캐쉬를 비움,cpu로 옮겼긴 한데 pytorch가 그걸 gpu가 캐싱하도록 만들수도 있다고 함. 그걸 메모리 효율을 위해 비움.

    
    # same input of first layer for fp model and quant model
    quant_inps = inps #이건 논문에서 본 것 같습니다. 블록단위 최적화 알고리즘에 사용할 Xq만들기. (nsamples, seqlen, hidden size)
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    

    attention_mask = masks

    # if attention_mask is not None: #하나의 샘플에서 얻은 마스크를 배치사이즈만큼 복사해서 쓰는데, 왜 이러는지는 모르겠다. gpt는 좀 간단하게 하려는 이유란다.
    #     attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    # else:
    #     logger.info(
    #         "No attention mask caught from the first layer."
    #         " Seems that model's attention works without a mask."
    #     )
    #     attention_mask_batch = None

    loss_func = torch.nn.MSELoss() #블록단위 최적화 알고리즘을 위한 손실함수
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume: #저장된 omni_parameter가 있으면 불러오고, 없으면 딕셔너리로 초기화
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        print("i : ", i)
        print("for문 시작때 i 확인\n")
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev) #layer가 각각 디코더 블록인가봄.
        
        if i==0:
            former_final_layernorm= None
        else:
            former_final_layernorm=layers[i-1].output.LayerNorm.to(dev)

        if "mixtral" in args.net.lower():  #일단 mixtral은 안쓸 것 같으니 제외하자.
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else: #여길 쓸듯? 아 여기부터 지랄이네. bertlayer도 decoderlayer라고 지정해버릴까
            if i==0:
                qlayer = DecoderLayer_1(lm.model.config, layer, args)
            elif i==len(layers)-1:
                qlayer = DecoderLayer_12(lm.model.config, layer, former_final_layernorm, args)
            else:
                qlayer = DecoderLayer_2to11(lm.model.config, layer, former_final_layernorm, args)

            # qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'): #amp. fp16연산을 활성화 해서 효율을 높인다는데, 모르겠다.
                    for j in range(args.nsamples): #nsample만큼. nsamples가 뭐지? 샘플수...? qlayer의 출력은 튜플임. 첫번째 원소가 hidden. 각 샘플당 첫번째 블록 출력 다 나옴.
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask[j].unsqueeze(0))[0]
                        if args.aug_loss: #추가적인 loss계산을 위해서 하나 더(?)라는데
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask[j].unsqueeze(0))[0]

        # init smooth parameters. weight는 수동으로 양자화 할 것이라네.
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift. 지금 보고 있는 layer의 출력 채널 크기와 같은 크기의 벡터를 qkt_smooth_scale이라는 파라미터로 추가!
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.attention.self.query.out_features,device=dev, dtype=dtype)))
            # qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            if i != 0:
                keypairs=pairs
            else:
                keypairs=pairs_for_firstblock
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    
                    for key in keypairs.keys(): #위에서 pairs라는 dict만들어 놓음. qlayer의 모든 모듈에서 이름하고 일치하는걸 loop을 돌리며 찾는다.
                        if key in name: #act_scales는 활성화값의 let용 스케일값을 저장하고 있는 dict이다. layer_name_prefix는 models.layer이다. 불러온 걸 gpu로 옮기고 최소값을 제한한다. 너무 작아지면 계산이 불안정하다네?
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            #활성화 값과 가중치 값의 비율을 사용해서 적절히 스케일 값을 조절한다네
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)

                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            #학습 가능한 파라미터로 등록.
                            qlayer.register_parameter(f"{keypairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{keypairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                            
        for name, param in qlayer.named_parameters():
            if param.requires_grad:  # requires_grad=True인 파라미터만 필터링
                print(f"Parameter name: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"Parameter value: {param}\n")
                print("-----------------------------------------------")

                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        
        if torch.eq(qlayer.fc1_smooth_scale, 0).any():
            print("fc1scale 값에 0이 포함되었다!!!!!\n")
        else:
            print("fc1scale 값에 0이 안 포함되었다!!!!!\n") #여기까지는 스무스 스케일에 0이 포함되지 않는다고 나옴

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training, 이렇게 float로 바꾸는 연산이라도, pytorch가 추적할 수도 있음. 그래서 no grad 쓴듯
            # create optimizer
            if torch.eq(qlayer.fc1_smooth_scale, 0).any():
                print(" qlayer.float()이후 fc1scale 값에 0이 포함되었다!!!!!\n")
            else:
                print("qlayer.float()이후 fc1scale 값에 0이 안 포함되었다!!!!!\n") #여기까지도 스무스 스케일에 0이 포함되지 않는다고 나옴
            

            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils_.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size): #한번에 한 배치들어갈테니 이정도 횟수가 시행될 것이다.
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        print("i=", i)
                        print("smooth 함수 바깥에서 i 확인\n")
                        smooth_and_quant_temporary(qlayer, args, is_llama, i) #앞에서 정한 스케일, 시프트 값으로 let수행하며 quant. 이때는 temp_weight사용하는 듯
                        
                        
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask[index:index+args.batch_size,])[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)


                        if args.aug_loss: # 이건 안함
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()): #손실 값이 유효한지 확인?
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu()) #손실 값을 리스트에 저장. 근데 detach는 왜하는 것일까?
                    optimizer.zero_grad() #그래디언트 초기화? 다음 계산에 누적해서 기울기가 계산되면 안된다고 함.
                    # 이 부분에서 옵티마지어가 일을 하네. loss_scaler가 amp 트레이닝때문에 쓰인다고 함. fp16이 불안정해서 무슨 보정을 해준다나 뭐라나.
                    

                    #이거 한번 써보자.
                    # loss.backward()
                    # optimizer.step()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu() #omni_parameter만 학습을 하는 것이다.
                    norm_list.append(norm.data) #계산한 그래디언트 노름 값을 리스트에 또 저장함.


                loss_mean = torch.stack(loss_list).mean() #이번 블록 계산에서 loss의 평균을 구함. torch.stack으로 여러 배치에서 계산된 것들을 하나의 텐서로 결합.
                norm_mean = torch.stack(norm_list).mean() #이번 블록 계산에서 norm의 평균을 구함. norm은 제곱 더하기 제곱을 루트 씌운 것과 비슷
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer) #temp weight와 bias 삭제
            del optimizer
        qlayer.half() #파이토치 메서드, 모든 파라미터와 버퍼를 fp16(반정밀도)로 변환
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama, i) #여태까지 학습한 omni parameter를 통해서 inplace quant 실행. 진짜 weight와 bias의 값이 바뀌고 양자화된다.
        if args.epochs>0: #에포크가 0보다 크다면 = 학습을 했다면 양자화된 모델의 입력을 업데이트함
            # update input of quantization model. 다음 블록에 집어넣을 입력들을 만들기 위해서 이 for loop을 수행함.
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        #여기선 quant_inps[j]라서 unsqueeze를 한 것. nsamples중 j번째. 그래서 맨 앞 배치차원에 해당하는 차원이 있어야 모델에 넣을 수 있음.
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask[j].unsqueeze(0))[0]
            #얘는 모르겠다. qlayer 속의 affinequnatizer들에서 scale과 round_zero_point를 다 삭제해버리고 거기에 s붙는 속성들을 만들어서 quantizer를 못 쓰게 만들어버린다.
            register_scales_and_zeros(qlayer)

            #여기에서 원래 layer가 qlayer로 바뀌는 것 같다.
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth")) #omni parameter들을 이 경로에다가 저장.
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        #이 부분은 실제 하드웨어상에서 quantization 수행하는 부분인가?
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales #아까 register_scales_and_zeros가 여기 쓰이는구나!
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache

    #모든 블록단위 최적화를 끝낸 모델을 return. 근데 main()을 보니까 따로 omniquant(...)를 받는 객체가 없네?
    return model

