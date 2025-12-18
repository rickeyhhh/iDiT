import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhaseQuantSTE(torch.autograd.Function):
    """Complex-Phase STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w_real, w_imag):
        phase = torch.angle(w_real + 1j * w_imag)
        
        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg
        
        s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
        s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
        
        s_re = torch.clamp(s_re, min=1e-6)
        s_im = torch.clamp(s_im, min=1e-6)
        
        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)
        
        qw_real[real_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_pos] = 1.0
        qw_imag[imag_neg] = -1.0
        
        qw_real_scaled = qw_real * s_re
        qw_imag_scaled = qw_imag * s_im
        
        return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

    @staticmethod
    def backward(ctx, grad_w_real, grad_w_imag):
        return grad_w_real, grad_w_imag

class PhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual quantization"""
    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real, error_imag)
        qw_real = qw_real_o1 + qw_real_o2
        qw_imag = qw_imag_o1 + qw_imag_o2
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class PhaseQuantSTE_V3(torch.autograd.Function):
    """Three-step residual quantization"""
    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

class PhaseQuantSTE_V4(torch.autograd.Function):
    """Four-step residual quantization"""
    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        error_real_3 = error_real_2 - qw_real_o3
        error_imag_3 = error_imag_2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = PhaseQuantSTE.apply(error_real_3, error_imag_3)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

@torch.no_grad()
def quantize_complex_tensor(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PhaseQuant logic to complex weight tensors"""
    phase = torch.angle(w_real + 1j * w_imag)

    real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
    real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
    imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
    imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

    mask_real = real_pos | real_neg
    mask_imag = imag_pos | imag_neg

    s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
    s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
    
    s_re = torch.clamp(s_re, min=1e-6)
    s_im = torch.clamp(s_im, min=1e-6)
    if torch.isnan(s_re) or torch.isinf(s_re): s_re = torch.tensor(1e-6, device=w_real.device)
    if torch.isnan(s_im) or torch.isinf(s_im): s_im = torch.tensor(1e-6, device=w_imag.device)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    
    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0

    qw_real_scaled = qw_real * s_re
    qw_imag_scaled = qw_imag * s_im
    return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

def apply_complex_inspired_quantization(model: nn.Module):
    """Apply complex-inspired quantization to real-valued model"""
    print("Applying complex-inspired quantization (PhaseQuant-based)...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        A = module.weight.data
        if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
            print(f"  -> Skipping layer (non-even dimensions): {A.shape}")
            return

        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        U_re_q, U_im_q = quantize_complex_tensor(U_re, U_im)
        W_re_q, W_im_q = quantize_complex_tensor(W_re, W_im)

        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q

        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        module.weight.data = A_quant.to(A.dtype)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("Complex-inspired quantization completed.")
    return model

class QATLinearComplexPhaseV1(nn.Linear):
    """Complex-Phase V1 QAT linear layer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV2(nn.Linear):
    """Complex-Phase V2 QAT linear layer (1-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V2.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V2.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV3(nn.Linear):
    """Complex-Phase V3 QAT linear layer (2-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V3.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V3.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV4(nn.Linear):
    """Complex-Phase V4 QAT linear layer (3-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V4.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V4.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

METHOD_MAP = {
    'complex_phase_v1': QATLinearComplexPhaseV1,
    'complex_phase_v2': QATLinearComplexPhaseV2,
    'complex_phase_v3': QATLinearComplexPhaseV3,
    'complex_phase_v4': QATLinearComplexPhaseV4,
}

def replace_modules_for_qat(model: nn.Module, method: str):
    """Recursively replace nn.Linear layers in the model with QAT layers"""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHOD_MAP.keys())}")

    TargetQATClass = METHOD_MAP[method]
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_modules_for_qat(module, method)
        
        if isinstance(module, nn.Linear):
            if module.in_features % 2 != 0 or module.out_features % 2 != 0:
                print(f"  -> Skipping Complex-Phase replacement (non-even dimensions): {name} ({module.in_features}, {module.out_features})")
                continue
            
            print(f"  -> Replacing layer: {name} with {TargetQATClass.__name__}")
            new_module = TargetQATClass(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device
            )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            
            setattr(model, name, new_module)