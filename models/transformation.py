
import torch
import pdb

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()#텐서 딥카피
        #truncated_tensor의 값 중, 절댓값이 threshold보다 작은 것들만 골라내는 boolean mask indexing. 
        #truncated_tensor.abs() < threshold 의 결과가 true와 false로만 이루어진 텐서이다.
        #.sign()은 골라낸 것의 부호를 값으로 가진 텐서 반환. 여기다가 threshold를 곱.
        #결과적으로 절댓값보다 작은 것들은 그것의 부호에다가 threshold를 곱한 것으로 교체된다.
        #너무 작은 절댓값을 가진 것들을 없애서 학습에서 발생하는 문제를 없애기 위함으로 보임. 양자화의 손실문제인가?
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


#let적용하는 부분 같은데.
def smooth_ln_fcs_temporary(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = True
    #fcs가 리스트가 아니면 리스트로 바꾸는 부분. 여러개의 fc들일수도 있으니
    if not isinstance(fcs, list):
        fcs = [fcs]
    #layernorm 모듈이 bias 속성을 가지고 있으며, 그 값이 None이 아닌지 확인하는 조건
    #바이어스가 있으면, let수식에 있는 변환 적용
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    #근데, 이러면 x에 scales 값을 곱해주어야 할 텐데. 그건 어디서 처리하는 것이지?
    #ln또는 fully connected에 let적용
    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts #@가 행렬곱이라는데?
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1,-1)


def smooth_fc_fc_temporary(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = True
    fc2.use_temporary_parameter = True
    if hasattr(fc1, 'temp_weight'): #이렇게 동적으로 속성 추가하는 것은 pytorch가 관리안해준다고 함. register_buffer 등과 다름.
        fc1.temp_bias = fc1.temp_bias - shifts
        fc1.temp_bias = fc1.temp_bias/scales.view(-1)
        fc1.temp_weight = fc1.temp_weight/scales.view(-1,1)
    else:
        fc1.temp_bias = fc1.bias/scales.view(-1)
        fc1.temp_weight = fc1.weight/scales.view(-1,1)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.temp_bias = fc2.bias + fc2.weight@shifts
    else:
        fc2.temp_bias = fc2.weight@shifts
    fc2.temp_weight = fc2.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True
    if hasattr(q_proj, 'temp_weight'): 
        q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    else:
        q_proj.temp_weight = q_proj.weight/scales.view(-1,1)

    if hasattr(q_proj, 'temp_bias'): 
        q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)
    else:
        q_proj.temp_bias = q_proj.bias/scales.view(-1)

    if hasattr(k_proj, 'temp_weight'): 
        k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    else:
        k_proj.temp_weight = k_proj.weight*scales.view(-1,1)

    if hasattr(k_proj, 'temp_bias'): 
        k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)
    else:
        k_proj.temp_bias = k_proj.bias*scales.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))


def smooth_fc_fc_inplace(fc1, fc2, scales,shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False
    fc1.bias.sub_(shifts)
    fc1.bias.div_(scales.view(-1))
    fc1.weight.div_(scales.view(-1,1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales,):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False
    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(scales.view(-1,1))
    k_proj.bias.mul_(scales.view(-1))