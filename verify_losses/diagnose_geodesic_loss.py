"""Diagnose why geodesic loss is stuck at ~24.6 in training.

Questions:
1. Is M_cross near-singular (all σ≈0)?
2. Is the gradient w.r.t. C_pred via SVD + stable_acos reasonable?
3. Is the gradient w.r.t. H via StableEigh backward reasonable?
"""
import torch
import sys
sys.path.insert(0, '/home/pepe/codebench/GrassD_sphnet_ver2')
from src.training.losses import StableEigh, stable_acos

torch.manual_seed(42)
nocc = 13
nbasis = 66  # def2-svp for ethanol

print("=" * 65)
print("  Diagnose geodesic loss gradient flow")
print("=" * 65)

# ---- Step 1: simulate predicted and target occupied MOs ----
# C_gt: ground truth occupied MOs (nbasis x nocc)
C_gt = torch.linalg.qr(torch.randn(nbasis, nocc))[0]

# C_pred: predicted occupied MOs (initially random)
C_pred = torch.linalg.qr(torch.randn(nbasis, nocc))[0]
C_pred.requires_grad_(True)

# Overlap matrix (identity in orthogonalized basis, but not strictly)
S = torch.eye(nbasis)

# ---- Step 2: compute M_cross and geodesic loss ----
M_cross = C_gt.T @ S @ C_pred  # (nocc x nocc)

_, sigma, _ = torch.linalg.svd(M_cross)
print(f"\n[Initial M_cross singular values]")
print(f"  sigma = {sigma.detach().numpy()}")
print(f"  min={sigma.min().item():.4f}, max={sigma.max().item():.4f}")

theta = stable_acos(sigma.clamp(-1, 1))
geodesic_loss = (theta ** 2).sum()
print(f"\n  geodesic_loss = {geodesic_loss.item():.4f}")
print(f"  max possible   = {nocc * (torch.pi / 2)**2:.4f}")
print(f"  ratio          = {geodesic_loss.item() / (nocc * (torch.pi / 2)**2):.2%}")

# ---- Step 3: check gradient flow through SVD + stable_acos ----
grad_c_pred, = torch.autograd.grad(geodesic_loss, C_pred, create_graph=True)
print(f"\n[Gradient through geodesic -> C_pred]")
print(f"  grad C_pred norm     = {grad_c_pred.norm().item():.4f}")
print(f"  grad finite?         = {torch.isfinite(grad_c_pred).all().item()}")
print(f"  grad C_pred[:4,:4]   = \n{grad_c_pred[:4, :4]}")

# ---- Step 4: simulate full chain through Hamiltonians ----
# Create a synthetic Hamiltonian whose eigenvectors approximate C_gt
# and a predicted Hamiltonian that's wrong
torch.manual_seed(123)
H_gt = torch.randn(nbasis, nbasis)
H_gt = H_gt @ H_gt.T  # symmetric
H_gt.requires_grad_(False)

H_pred = torch.randn(nbasis, nbasis)
H_pred = H_pred @ H_pred.T
H_pred.requires_grad_(True)

# Solve eigh with StableEigh
vals_pred, vecs_pred = StableEigh.apply(H_pred, 3.0, nocc)
vecs_pred_occ = vecs_pred[:, :nocc]

# M_cross from full chain
M_cross_full = C_gt.T @ S @ vecs_pred_occ
_, sigma_full, _ = torch.linalg.svd(M_cross_full)
theta_full = stable_acos(sigma_full.clamp(-1, 1))
loss_full = (theta_full ** 2).sum()

print(f"\n[Full chain: H_pred -> eigh -> occ -> SVD -> geodesic]")
print(f"  geodesic_loss from H_pred = {loss_full.item():.4f}")

# Check gradient flow
grad_H, = torch.autograd.grad(loss_full, H_pred, create_graph=True)
print(f"  grad H_pred norm        = {grad_H.norm().item():.4f}")
print(f"  grad H_pred finite?     = {torch.isfinite(grad_H).all().item()}")
print(f"  grad H_pred[:4,:4]      = \n{grad_H[:4, :4]}")

# ---- Step 5: check if StableEigh backward kills gradients ----
# Compare gradient with vs without nocc
H_pred2 = H_pred.detach().clone().requires_grad_(True)
vals2, vecs2 = StableEigh.apply(H_pred2, 3.0, -1)  # no VV truncation
vecs2_occ = vecs2[:, :nocc]
M2 = C_gt.T @ S @ vecs2_occ
_, s2, _ = torch.linalg.svd(M2)
t2 = stable_acos(s2.clamp(-1, 1))
l2 = (t2 ** 2).sum()
grad_H2, = torch.autograd.grad(l2, H_pred2)

diff = (grad_H - grad_H2).norm().item() / grad_H.norm().item()
print(f"\n[VV truncation effect]")
print(f"  relative grad diff (nocc=-1 vs nocc={nocc}) = {diff:.4f}")

print("\n" + "=" * 65)
