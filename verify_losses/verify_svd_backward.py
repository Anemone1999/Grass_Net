"""Check if torch.linalg.svd backward produces NaN when singular values are near-degenerate.

This simulates the geodesic loss: M_cross = C_gt.T @ S @ C_pred.
When predictions are bad (random), all singular values are small and close -> SVD backward may NaN.
"""
import torch
import sys
sys.path.insert(0, '/home/pepe/codebench/GrassD_sphnet_ver2')
from src.training.losses import stable_acos

torch.manual_seed(42)

nocc = 13  # ethanol occupied orbitals
n_test = 100

print("=" * 60)
print("  SVD backward stability test (simulating geodesic loss)")
print("=" * 60)

for scenario, sigma_mean in [("good prediction (all σ≈1)", 0.99),
                              ("medium (mixed)", 0.5),
                              ("bad prediction (all σ≈0)", 0.01)]:
    nan_count = 0
    inf_count = 0
    finite_count = 0
    grad_values = []

    for _ in range(n_test):
        # Simulate M_cross with controlled singular values
        U = torch.linalg.qr(torch.randn(nocc, nocc))[0]
        V = torch.linalg.qr(torch.randn(nocc, nocc))[0]
        s = torch.rand(nocc) * 0.1 + sigma_mean  # tight cluster around sigma_mean
        # Ensure some are very close -> degenerate SVD
        if scenario == "bad prediction (all σ≈0)":
            s = s * 0.01 + 0.005  # all ~0-0.015, very degenerate

        M = U @ torch.diag(s) @ V.T
        M.requires_grad_(True)

        # Forward: geodesic loss
        _, sigma, _ = torch.linalg.svd(M)
        theta = stable_acos(sigma.clamp(-1, 1))
        loss = (theta ** 2).sum()

        # Backward
        try:
            grad, = torch.autograd.grad(loss, M, create_graph=False)
            if torch.isnan(grad).any():
                nan_count += 1
            elif torch.isinf(grad).any():
                inf_count += 1
            else:
                finite_count += 1
                grad_values.append(grad.abs().max().item())
        except RuntimeError as e:
            if "NaN" in str(e) or "nan" in str(e):
                nan_count += 1
            else:
                raise

    print(f"\n[{scenario}]")
    print(f"  NaN gradients : {nan_count}/{n_test}")
    print(f"  Inf gradients : {inf_count}/{n_test}")
    print(f"  Finite        : {finite_count}/{n_test}")
    if grad_values:
        print(f"  |grad|_max     : min={min(grad_values):.2e}, median={sorted(grad_values)[len(grad_values)//2]:.2e}")

print("\n" + "=" * 60)

# Bonus: check if the gradient direction makes sense
print("\n  [Sanity check: gradient direction]")
print("  Random vs true M_cross should push σ towards 1")
M_init = torch.randn(nocc, nocc) * 0.1
M_init = M_init @ M_init.T  # sym
M_init = M_init / M_init.norm()
M_init.requires_grad_(True)

_, sigma_init, _ = torch.linalg.svd(M_init)
theta = stable_acos(sigma_init.detach().clamp(-1, 1))
loss = (theta ** 2).sum()
grad, = torch.autograd.grad(loss, M_init)
print(f"  sigma_init    : {sigma_init.detach().numpy()}")
print(f"  grad M norm   : {grad.norm().item():.4f}")
print(f"  grad finite?  : {torch.isfinite(grad).all().item()}")
