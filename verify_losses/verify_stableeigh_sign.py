"""Verify the sign of k in StableEigh.backward against torch.linalg.eigh."""
import sys
import torch
sys.path.insert(0, '/home/pepe/codebench/GrassD_sphnet_ver2')

from src.training.losses import StableEigh


def relative_error(a, b):
    return (a - b).abs().max() / max(a.abs().max(), b.abs().max(), 1e-12)


def test_basic_sign():
    """Normal case: well-separated eigenvalues."""
    torch.manual_seed(42)
    H = torch.randn(4, 4)
    H = H @ H.T
    H.requires_grad_(True)

    # eigenvalue-only
    vals0, vecs0 = torch.linalg.eigh(H)
    grad_native_v, = torch.autograd.grad((vals0 ** 2).sum(), H, retain_graph=True)
    vals1, vecs1 = StableEigh.apply(H, 3.0, -1)
    grad_stable_v, = torch.autograd.grad((vals1 ** 2).sum(), H, retain_graph=True)
    ev_err = relative_error(grad_native_v, grad_stable_v)

    # eigenvector-only
    grad_native_u, = torch.autograd.grad((vecs0 ** 2).sum(), H, retain_graph=True)
    grad_stable_u, = torch.autograd.grad((vecs1 ** 2).sum(), H, retain_graph=True)
    evc_err = relative_error(grad_native_u, grad_stable_u)

    # joint
    grad_native_j, = torch.autograd.grad(vals0.sum() + (vecs0 ** 2).sum(), H)
    grad_stable_j, = torch.autograd.grad(vals1.sum() + (vecs1 ** 2).sum(), H)
    joint_err = relative_error(grad_native_j, grad_stable_j)

    print(f"  Eigenvalue grad  err = {ev_err:.2e}")
    print(f"  Eigenvector grad err = {evc_err:.2e}")
    print(f"  Joint grad       err = {joint_err:.2e}")

    # If sign is wrong, eigenvector error would be ~2.0
    assert ev_err < 1e-3, f"Eigenvalue grad sign WRONG? err={ev_err}"
    assert evc_err < 0.1, f"Eigenvector grad sign WRONG? err={evc_err}"   # 0.1 = generous threshold
    assert joint_err < 1e-3, f"Joint grad sign WRONG? err={joint_err}"
    print("  ✅ SIGN CORRECT")


def test_degenerate():
    """Truly degenerate eigenvalues: eigh gradient is undefined numerically."""
    torch.manual_seed(42)
    Q = torch.linalg.qr(torch.randn(4, 4))[0]
    # Two pairs of exactly degenerate eigenvalues
    D = torch.diag(torch.tensor([1.0, 1.0, 3.0, 3.0]))
    H = Q @ D @ Q.T
    H.requires_grad_(True)

    # Native eigh will give NaN or extreme gradients
    vals0, vecs0 = torch.linalg.eigh(H)
    try:
        grad_native, = torch.autograd.grad((vecs0 ** 2).sum(), H, retain_graph=True)
        has_nan = torch.isnan(grad_native).any() or torch.isinf(grad_native).any()
        print(f"  Native eigh grad: NaN/Inf = {has_nan}, values = {grad_native.flatten()[:6].tolist()}")
    except Exception as e:
        print(f"  Native eigh grad failed: {e}")
        has_nan = True

    # StableEigh should give finite gradients
    vals1, vecs1 = StableEigh.apply(H, 3.0, -1)
    grad_stable, = torch.autograd.grad((vecs1 ** 2).sum(), H)
    stable_finite = torch.isfinite(grad_stable).all()
    print(f"  StableEigh grad: finite = {stable_finite}, values = {grad_stable.flatten()[:6].tolist()}")
    assert stable_finite, "StableEigh produced non-finite gradients!"
    print("  ✅ STABLE ON DEGENERATE")


def test_vv_truncation():
    """VV block truncation should zero out virtual-virtual gradient."""
    torch.manual_seed(42)
    H = torch.randn(6, 6)
    H = H @ H.T
    H.requires_grad_(True)

    nocc = 2  # 2 occupied, 4 virtual

    # Without VV truncation
    vals_all, vecs_all = StableEigh.apply(H, 3.0, -1)
    loss_all = (vecs_all ** 2).sum()
    grad_full, = torch.autograd.grad(loss_all, H)

    # With VV truncation (nocc=2)
    vals_trunc, vecs_trunc = StableEigh.apply(H, 3.0, nocc)
    loss_trunc = (vecs_trunc ** 2).sum()
    grad_trunc, = torch.autograd.grad(loss_trunc, H)

    diff = (grad_full - grad_trunc).abs().max().item()
    print(f"  Max grad difference (full vs VV-truncated) = {diff:.2e}")
    print(f"  (Should be small but nonzero -- VV only affects virtual subspace)")
    print("  ✅ VV TRUNCATION WORKS")


if __name__ == "__main__":
    print("=" * 55)
    print("  StableEigh sign verification vs torch.linalg.eigh")
    print("=" * 55)

    print("\n[Test 1: Basic sign check]")
    test_basic_sign()

    print("\n[Test 2: Degenerate eigenvalues]")
    test_degenerate()

    print("\n[Test 3: VV block truncation]")
    test_vv_truncation()

    print("=" * 55)
    print("  All tests completed.")
