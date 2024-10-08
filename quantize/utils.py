from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

#affine quantizer도 일종의 모듈인가보네. affine quantizer안에 있는 lwc 관련 파라미터, quantlinear들 안에 있는 let파라미터들을 다 dict로 모음.
def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama, i):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
                
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        elif i!=0:
            print(i)
            print("smoothandquant_temp의 i확인\n")
            smooth_ln_fcs_temporary(model.attention.self.former_final_layernorm,[model.attention.self.query, model.attention.self.key, model.attention.self.value],
                                    model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.attention.output.LayerNorm,[model.intermediate.dense],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.attention.self.value,model.attention.output.dense,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.attention.self.query, model.attention.self.key,
                                model.qkt_smooth_scale)
            model.output.dense.temp_weight = model.output.dense.weight
        else: #1번 layer의 경우에는 former layernorm이 없다. 
            # smooth_ln_fcs_temporary(model.attention.self.former_final_layernorm,[model.attention.self.query, model.attention.self.key, model.attention.self.value],
            #                         model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.attention.output.LayerNorm,[model.intermediate.dense],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.attention.self.value,model.attention.output.dense,
                                model.out_smooth_scale, model.out_smooth_shift)
            if torch.isnan(model.attention.self.value.temp_weight).any():
                print("ln.temp_weight에 NaN 값이 포함되어 있습니다.")
            else:
                print("ln.temp_weight에는 NaN 값이 없습니다.")
            smooth_q_k_temporary(model.attention.self.query, model.attention.self.key,
                                model.qkt_smooth_scale)
            model.output.dense.temp_weight = model.output.dense.weight


        # else:
        #     smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
        #                             model.qkv_smooth_scale,model.qkv_smooth_shift)
        #     smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
        #                             model.fc1_smooth_scale,model.fc1_smooth_shift)
        #     smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
        #                         model.out_smooth_scale, model.out_smooth_shift)
        #     smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
        #                         model.qkt_smooth_scale)
        #     model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight


    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True

            if torch.isnan(module.temp_weight).any():
                print(name)
                print("temp_weight에 NaN 값이 포함되어 있습니다.\n")
            else:
                print(name)
                print("temp_weight에 NaN 값이 없습니다.\n")



            
def clear_temp_variable(model): #named_modules 중에서 quantlinear 타입들만 찾아서 temp weight와 temp bias 싹다 삭제시켜버림.
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama, i): #각 모듈들의 Quantlinear 모듈들의 weight와 bias를 모두 inplace 연산으로, let시킴. & 양자화까지
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        elif i !=0:
            smooth_ln_fcs_inplace(model.attention.self.former_final_layernorm,[model.attention.self.query, model.attention.self.key, model.attention.self.value],
                                    model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.attention.output.LayerNorm,[model.intermediate.dense],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.attention.self.value,model.attention.output.dense,
                                model.out_smooth_scale, model.out_smooth_shift)
        
        else: #1번 layer는 former layernorm이 없다.
            # smooth_ln_fcs_inplace(model.attention.self.former_final_layernorm,[model.attention.self.query, model.attention.self.key, model.attention.self.value],
            #                         model.qkv_smooth_scale, model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.attention.output.LayerNorm,[model.intermediate.dense],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.attention.self.value,model.attention.output.dense,
                                model.out_smooth_scale, model.out_smooth_shift)

        smooth_q_k_inplace(model.attention.self.query, model.attention.self.key,
                                model.qkt_smooth_scale)
        # else: # opt
        #     smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
        #                             model.qkv_smooth_scale,model.qkv_smooth_shift)
        #     smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
        #                             model.fc1_smooth_scale,model.fc1_smooth_shift)
        #     smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
        #                         model.out_smooth_scale, model.out_smooth_shift)

        # smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
        #                     model.qkt_smooth_scale)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
