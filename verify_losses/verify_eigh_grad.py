"""Verify StableEigh gradient correctness vs native torch.linalg.eigh.

Tests:
  A: Random matrices - forward (E, C) and gradient direction
  B: Real Fock matrices from dataset - gradient for grassmann losses
  C: K matrix sign and truncation accuracy
  D: Impact of near-degenerate eigenvalues
"""
import sys, os, re, pickle
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
DEVICE = 'cpu'

sys.path.insert(0, '/home/pepe/codebench/GrassD_sphnet_ver2')
from src.training.losses import StableEigh, stable_acos

# ============================================================
# Helpers
# ============================================================
def cos_sim(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return 1.0
    a_f, b_f = a.flatten(), b.flatten()
    na, nb = a_f.norm(), b_f.norm()
    if na < 1e-12 and nb < 1e-12:
        return 1.0
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return (a_f @ b_f).item() / (na * nb)

def solve_eigh_native(H, S, eps=1e-8):
    """Standard eigh with overlap orthogonalization (no gradient tricks)."""
    s_vals, s_vecs = torch.linalg.eigh(S)
    s_vals = torch.clamp(s_vals, min=eps)
    X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
    Fs = X.T @ H @ X
    e_vals, e_vecs = torch.linalg.eigh(Fs)
    C = X @ e_vecs
    return e_vals, C, Fs, X

def solve_eigh_stable(H, S, nocc, trunc_factor=3.0, eps=1e-8):
    """Same as _solve_eigh in losses.py: uses StableEigh."""
    s_vals, s_vecs = torch.linalg.eigh(S)
    s_vals = torch.clamp(s_vals, min=eps)
    X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
    Fs = X.T @ H @ X
    e_vals, e_vecs = StableEigh.apply(Fs, trunc_factor, nocc)
    C = X @ e_vecs
    return e_vals, C, Fs, X

# ============================================================
# Test A: Random matrices (statistical)
# ============================================================
def test_random_matrices():
    print("=" * 65)
    print("  Test A: Random matrices - gradient direction")
    print("=" * 65)

    for size, label in [(4, "4×4 (tiny)"), (20, "20×20 (small)"), (66, "66×66 (ethanol)")]:
        nocc = size // 3
        cos_list = []
        for seed in range(200):
            torch.manual_seed(seed)
            H = torch.randn(size, size)
            H = H @ H.T
            H.requires_grad_(True)
            S = torch.eye(size)
            _, C0, Fs0, X0 = solve_eigh_native(H, S)
            _, C1, Fs1, X1 = solve_eigh_stable(H, S, nocc, 3.0)

            # Loss: sum(C_occ²) — grassmann-agnostic baseline
            loss0 = (C0[:, :nocc] ** 2).sum()
            loss1 = (C1[:, :nocc] ** 2).sum()
            g0, = torch.autograd.grad(loss0, H, retain_graph=True)
            g1, = torch.autograd.grad(loss1, H)
            cos_list.append(cos_sim(g0, g1))

        cos_t = torch.tensor(cos_list)
        print(f"\n  {label}  nocc={nocc}  N=200")
        print(f"    cos_sim:  mean={cos_t.mean():.6f}  min={cos_t.min():.6f}  "
              f"<0.99={float((cos_t < 0.99).float().mean())*100:.1f}%")
        if cos_t.min() > 0.99:
            print(f"    ✅ ALL cos > 0.99")
        else:
            worst = cos_t.argmin()
            print(f"    ❌ worst seed={worst.item()}, cos={cos_t.min():.6f}")

# ============================================================
# Test B: Real Fock matrices from dataset
# ============================================================
def load_real_samples(n_mol=8):
    DATASET = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/data.mdb"
    SPLIT = "/home/pepe/workbench/basisset_scaling/lmdb_data/sphnet_gcnet_ethanol/def2svp/MD17_ethanol_trainset_with_5000_data.pt"
    import lmdb
    import pyscf.gto as gto

    env = lmdb.open(DATASET, readonly=True, lock=False, readahead=False, meminit=False)
    split = torch.load(SPLIT, map_location='cpu', weights_only=False)
    test_ids = split[2].tolist()[:n_mol]
    samples = []
    with env.begin() as txn:
        for abs_idx in test_ids:
            raw = pickle.loads(txn.get(int(abs_idx).to_bytes(4, 'big')))
            atoms = np.frombuffer(raw['atoms'], dtype=np.int32)
            pos = np.frombuffer(raw['pos'], dtype=np.float64).reshape(len(atoms), 3)
            H_flat = np.frombuffer(raw['Ham'], dtype=np.float64)
            H_init = raw['Ham_init']
            if isinstance(H_init, bytes):
                H_init = np.frombuffer(H_init, dtype=np.float64)
            norbs = H_init.shape[0] if H_init.ndim == 2 else int(np.sqrt(H_init.size))
            H = H_flat.reshape(norbs, norbs)
            mol = gto.M(atom=[(int(z), pos[i]) for i, z in enumerate(atoms)],
                        basis='def2-svp', unit='ang', verbose=0)
            s1e = mol.intor('int1e_ovlp')
            nocc = int(np.sum(atoms)) // 2
            samples.append({'H': H, 'H_init': H_init, 'S': s1e, 'nocc': nocc, 'norb': norbs,
                            'atoms': atoms, 'idx': abs_idx})
    env.close()
    return samples

def test_real_fock_matrices():
    print("\n" + "=" * 65)
    print("  Test B: Real Fock matrices from ethanol dataset")
    print("=" * 65)

    samples = load_real_samples(n_mol=8)
    print(f"  Loaded {len(samples)} molecules")

    # --- Test 1: sum(C_occ²) on all molecules ---
    print("\n  --- loss: sum(C_occ²) ---")
    for si, s in enumerate(samples):
        H_ref = torch.tensor(s['H'] - s['H_init'], dtype=torch.float64)
        S = torch.tensor(s['S'], dtype=torch.float64)
        nocc = s['nocc']

        H0 = H_ref.clone().requires_grad_(True)
        _, C0, _, _ = solve_eigh_native(H0, S)
        l0 = (C0[:, :nocc] ** 2).sum()
        g_nat, = torch.autograd.grad(l0, H0, retain_graph=True)

        H1 = H_ref.clone().requires_grad_(True)
        _, C1, _, _ = solve_eigh_stable(H1, S, nocc, 3.0)
        l1 = (C1[:, :nocc] ** 2).sum()
        g_sta, = torch.autograd.grad(l1, H1)

        c = cos_sim(g_nat, g_sta)
        print(f"    mol[{si}]  nbf={s['norb']}  nocc={nocc}  "
              f"cos(g_nat, g_sta)={c:.6f}  {'✅' if c > 0.99 else '❌'}")

    # --- Test 2: Full grassmann losses on first molecule ---
    print("\n  --- Full grassmann losses (first molecule) ---")
    s = samples[0]
    H_ref = torch.tensor(s['H'] - s['H_init'], dtype=torch.float64)
    S = torch.tensor(s['S'], dtype=torch.float64)
    nocc = s['nocc']

    with torch.no_grad():
        _, C_gt, _, _ = solve_eigh_native(torch.tensor(s['H'], dtype=torch.float64), S)
    C_gt_occ = C_gt[:, :nocc].detach()

    # Projection
    H0 = H_ref.clone().requires_grad_(True)
    _, C0, _, _ = solve_eigh_native(H0, S)
    M0 = C_gt_occ.T @ S @ C0[:, :nocc]
    l0 = nocc - torch.linalg.matrix_norm(M0, ord='fro') ** 2
    g_nat, = torch.autograd.grad(l0, H0, retain_graph=True)

    H1 = H_ref.clone().requires_grad_(True)
    _, C1, _, _ = solve_eigh_stable(H1, S, nocc, 3.0)
    M1 = C_gt_occ.T @ S @ C1[:, :nocc]
    l1 = nocc - torch.linalg.matrix_norm(M1, ord='fro') ** 2
    g_sta, = torch.autograd.grad(l1, H1)

    c_proj = cos_sim(g_nat, g_sta)
    print(f"    projection:  cos(g_nat, g_sta)={c_proj:.6f}  {'✅' if c_proj > 0.99 else '❌'}")

    # Geodesic
    H0 = H_ref.clone().requires_grad_(True)
    _, C0, _, _ = solve_eigh_native(H0, S)
    M0 = C_gt_occ.T @ S @ C0[:, :nocc]
    _, s0, _ = torch.linalg.svd(M0)
    t0 = stable_acos(s0.clamp(-1, 1))
    l0 = (t0 ** 2).sum()
    g_nat, = torch.autograd.grad(l0, H0, retain_graph=True)

    H1 = H_ref.clone().requires_grad_(True)
    _, C1, _, _ = solve_eigh_stable(H1, S, nocc, 3.0)
    M1 = C_gt_occ.T @ S @ C1[:, :nocc]
    _, s1, _ = torch.linalg.svd(M1)
    t1 = stable_acos(s1.clamp(-1, 1))
    l1 = (t1 ** 2).sum()
    g_sta, = torch.autograd.grad(l1, H1)

    c_geo = cos_sim(g_nat, g_sta)
    print(f"    geodesic:     cos(g_nat, g_sta)={c_geo:.6f}  {'✅' if c_geo > 0.99 else '❌'}")

# ============================================================
# Test C: K matrix sign and truncation
# ============================================================
def test_k_matrix():
    print("\n" + "=" * 65)
    print("  Test C: K matrix sign verification")
    print("=" * 65)

    for size, label in [(4, "4×4"), (20, "20×20"), (66, "66×66")]:
        for seed in [0, 1, 2, 42, 99]:
            torch.manual_seed(seed)
            H = torch.randn(size, size)
            H = H @ H.T
            S = torch.eye(size)
            nocc = size // 3

            s_vals, s_vecs = torch.linalg.eigh(S)
            X = s_vecs / torch.sqrt(s_vals.clamp(min=1e-8)).unsqueeze(-2)
            Fs = X.T @ H @ X

            # Get eigenvalues for K computation
            E_nat, _ = torch.linalg.eigh(Fs)
            E_sta, _ = StableEigh.apply(Fs.clone(), 3.0, nocc)

            # Native K
            E_diff = E_nat.unsqueeze(-1) - E_nat.unsqueeze(-2)
            K_nat = torch.where(
                torch.abs(E_diff) < 1e-10,
                torch.zeros_like(E_diff),
                1.0 / E_diff
            )

            # Stable K (replicate the logic)
            E_diff_s = E_sta.unsqueeze(-1) - E_sta.unsqueeze(-2)
            eps_s = 10.0 ** (-3.0)
            valid_mask = torch.abs(E_diff_s) > eps_s
            K_sta = torch.where(valid_mask, 1.0 / E_diff_s, torch.zeros_like(E_diff_s))

            # VV mask
            N_bf = size
            idx = torch.arange(N_bf)
            virt_mask = (idx >= nocc).unsqueeze(-1) & (idx >= nocc).unsqueeze(-2)
            K_sta = torch.where(virt_mask, torch.zeros_like(K_sta), K_sta)

            # Compare: for non-truncated entries, sign must match
            nonzero = (K_nat.abs() > 1e-6) & (valid_mask & ~virt_mask)
            sign_match = (K_sta[nonzero].sign() == K_nat[nonzero].sign()).all().item()
            k_max = K_sta.abs().max().item()
            
            if seed == 42:
                vv_zeroed = (K_sta[virt_mask].abs() > 0).sum().item()
                print(f"  {label:8s} seed={seed:3d} sign_match={sign_match}  "
                      f"K_max={k_max:.1f}  VV_nonzero_ops={vv_zeroed}")

    print("\n  ✅ K matrix sign check complete")

# ============================================================
# Test D: Near-degenerate eigenvalue stability
# ============================================================
def test_degenerate():
    print("\n" + "=" * 65)
    print("  Test D: Near-degenerate eigenvalue stability")
    print("=" * 65)

    for gap in [1e-2, 1e-3, 1e-4, 1e-5]:
        for size in [10, 20, 66]:
            torch.manual_seed(42)
            Q = torch.linalg.qr(torch.randn(size, size))[0]
            # Create eigenvalue spectrum with one near-degenerate pair
            evals = torch.linspace(-2, 2, size)
            evals[size//2] = evals[size//2 - 1] + gap  # make a close pair
            
            # Ensure eigenvalues are sorted (needed for eigh comparison)
            evals = torch.sort(evals)[0]
            if evals[size//2] - evals[size//2 - 1] > gap * 1.1:
                # Swap to make the close pair adjacent
                pass

            H = Q @ torch.diag(evals) @ Q.T
            H.requires_grad_(True)
            S = torch.eye(size)
            nocc = size // 3

            # Native: may have issues
            e0, C0 = torch.linalg.eigh(H)
            loss0 = (C0[:, :nocc] ** 2).sum()
            g_nat = None
            nat_ok = False
            try:
                g_nat, = torch.autograd.grad(loss0, H, retain_graph=True)
                nat_ok = torch.isfinite(g_nat).all().item() and g_nat.abs().max().item() < 1e6
            except Exception:
                pass

            # Stable: should always work
            e1, C1 = StableEigh.apply(H, 3.0, nocc)
            loss1 = (C1[:, :nocc] ** 2).sum()
            g_sta, = torch.autograd.grad(loss1, H)
            sta_ok = torch.isfinite(g_sta).all().item()

            # Check direction if both ok
            c = cos_sim(g_nat, g_sta) if (nat_ok and sta_ok) else -1.0

            print(f"  size={size:3d}  gap={gap:.0e}  "
                  f"native_ok={'✅' if nat_ok else '❌'}  "
                  f"stable_ok={'✅' if sta_ok else '❌'}  "
                  f"cos={c:.4f}" if c >= 0 else
                  f"  size={size:3d}  gap={gap:.0e}  "
                  f"native_ok={'✅' if nat_ok else '❌'}  "
                  f"stable_ok={'✅' if sta_ok else '❌'}  "
                  f"cos=N/A")

# ============================================================
# Test E: Compare two fix approaches (transpose vs swap E_diff)
# ============================================================
def test_fix_equivalence():
    print("\n" + "=" * 65)
    print("  Test E: .T vs swap E_diff — equivalence check")
    print("=" * 65)

    class StableEigh_Transpose(torch.autograd.Function):
        """Fix A: delta = K * C_T_grad_C.T  (K = 1/(E_i-E_j))"""
        @staticmethod
        def forward(ctx, Hp, trunc, nocc_v):
            Eo, Vo = torch.linalg.eigh(Hp)
            ctx.save_for_backward(Eo, Vo)
            ctx.trunc = trunc; ctx.nocc_v = nocc_v
            return Eo, Vo
        @staticmethod
        def backward(ctx, gE, gC):
            E, V = ctx.saved_tensors
            K = 1.0 / (E.unsqueeze(-1) - E.unsqueeze(-2))
            K.diagonal(dim1=-1, dim2=-2).zero_()
            if ctx.nocc_v > 0 and ctx.nocc_v < E.shape[-1]:
                idx = torch.arange(E.shape[-1], device=E.device)
                virt = (idx >= ctx.nocc_v).unsqueeze(-1) & (idx >= ctx.nocc_v).unsqueeze(-2)
                K = torch.where(virt, torch.zeros_like(K), K)
            C_T_grad_C = V.T @ gC
            dH = K * C_T_grad_C.T + torch.diag_embed(gE)
            gH = V @ dH @ V.T
            return (gH + gH.T) * 0.5, None, None

    class StableEigh_SwapE(torch.autograd.Function):
        """Fix B: delta = K * C_T_grad_C  (K = 1/(E_j-E_i))"""
        @staticmethod
        def forward(ctx, Hp, trunc, nocc_v):
            Eo, Vo = torch.linalg.eigh(Hp)
            ctx.save_for_backward(Eo, Vo)
            ctx.trunc = trunc; ctx.nocc_v = nocc_v
            return Eo, Vo
        @staticmethod
        def backward(ctx, gE, gC):
            E, V = ctx.saved_tensors
            # swap: E_diff = E_j - E_i  instead of E_i - E_j
            K = 1.0 / (E.unsqueeze(-2) - E.unsqueeze(-1))
            K.diagonal(dim1=-1, dim2=-2).zero_()
            if ctx.nocc_v > 0 and ctx.nocc_v < E.shape[-1]:
                idx = torch.arange(E.shape[-1], device=E.device)
                virt = (idx >= ctx.nocc_v).unsqueeze(-1) & (idx >= ctx.nocc_v).unsqueeze(-2)
                K = torch.where(virt, torch.zeros_like(K), K)
            C_T_grad_C = V.T @ gC
            dH = K * C_T_grad_C + torch.diag_embed(gE)
            gH = V @ dH @ V.T
            return (gH + gH.T) * 0.5, None, None

    class StableEigh_OrigBug(torch.autograd.Function):
        """Original buggy: delta = K * C_T_grad_C  (K = 1/(E_i-E_j))"""
        @staticmethod
        def forward(ctx, Hp, trunc, nocc_v):
            Eo, Vo = torch.linalg.eigh(Hp)
            ctx.save_for_backward(Eo, Vo)
            ctx.trunc = trunc; ctx.nocc_v = nocc_v
            return Eo, Vo
        @staticmethod
        def backward(ctx, gE, gC):
            E, V = ctx.saved_tensors
            K = 1.0 / (E.unsqueeze(-1) - E.unsqueeze(-2))
            K.diagonal(dim1=-1, dim2=-2).zero_()
            if ctx.nocc_v > 0 and ctx.nocc_v < E.shape[-1]:
                idx = torch.arange(E.shape[-1], device=E.device)
                virt = (idx >= ctx.nocc_v).unsqueeze(-1) & (idx >= ctx.nocc_v).unsqueeze(-2)
                K = torch.where(virt, torch.zeros_like(K), K)
            C_T_grad_C = V.T @ gC
            dH = K * C_T_grad_C + torch.diag_embed(gE)
            gH = V @ dH @ V.T
            return (gH + gH.T) * 0.5, None, None

    methods = [
        ("Fix A (K*C.T)", StableEigh_Transpose),
        ("Fix B (swap E)", StableEigh_SwapE),
        ("Orig Bug (K*C)", StableEigh_OrigBug),
    ]

    # --- Subtest 1: Random matrices ---
    print("\n  --- Subtest 1: Random matrices (N=100) ---")
    for size, label in [(10, "10×10"), (66, "66×66")]:
        nocc = size // 3
        print(f"\n  {label}  nocc={nocc}:")
        for mname, mcls in methods:
            allclose_count = 0
            cossim_list = []
            for seed in range(100):
                torch.manual_seed(seed)
                H = torch.randn(size, size); H = H @ H.T
                S = torch.eye(size)

                # Native
                H0 = H.clone().requires_grad_(True)
                E0, C0 = torch.linalg.eigh(H0)
                l0 = (C0[:, :nocc] ** 2).sum()
                g_nat, = torch.autograd.grad(l0, H0, retain_graph=True)

                # Method
                H1 = H.clone().requires_grad_(True)
                E1, C1 = mcls.apply(H1, 3.0, nocc)
                l1 = (C1[:, :nocc] ** 2).sum()
                g_mtd, = torch.autograd.grad(l1, H1)

                c = cos_sim(g_nat, g_mtd)
                cossim_list.append(c)
                if torch.allclose(g_nat, g_mtd, atol=1e-6):
                    allclose_count += 1

            ct = torch.tensor(cossim_list)
            print(f"    {mname:20s}  cos_mean={ct.mean():.6f}  cos_min={ct.min():.6f}  "
                  f"allclose={allclose_count}/100")

    # --- Subtest 2: Real Fock matrices ---
    print("\n  --- Subtest 2: Real Fock matrices (8 mol) ---")
    samples = load_real_samples(n_mol=8)
    print(f"  Loaded {len(samples)} molecules")
    for mname, mcls in methods:
        results = []
        for si, s in enumerate(samples):
            H_ref = torch.tensor(s['H'] - s['H_init'], dtype=torch.float64)
            S = torch.tensor(s['S'], dtype=torch.float64)
            nocc = s['nocc']

            # Native
            H0 = H_ref.clone().requires_grad_(True)
            Fs0 = (lambda X: X.T @ H0 @ X)(
                torch.linalg.eigh(S)[1] / torch.sqrt(torch.clamp(torch.linalg.eigh(S)[0], min=1e-8)).unsqueeze(-2)
            ) if False else H0  # no orthog for clean comparison
            # Actually use full orthog
            s_vals, s_vecs = torch.linalg.eigh(S)
            s_vals = torch.clamp(s_vals, min=1e-8)
            X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)

            H0 = H_ref.clone().requires_grad_(True)
            Fs0 = X.T @ H0 @ X
            E0, C0 = torch.linalg.eigh(Fs0)
            C0 = X @ C0
            l0 = (C0[:, :nocc] ** 2).sum()
            g_nat, = torch.autograd.grad(l0, H0, retain_graph=True)

            H1 = H_ref.clone().requires_grad_(True)
            Fs1 = X.T @ H1 @ X
            E1, V1 = mcls.apply(Fs1, 3.0, nocc)
            C1 = X @ V1
            l1 = (C1[:, :nocc] ** 2).sum()
            g_mtd, = torch.autograd.grad(l1, H1)

            c = cos_sim(g_nat, g_mtd)
            match = torch.allclose(g_nat, g_mtd, atol=1e-6)
            results.append((c, match))

        cos_vals = [r[0] for r in results]
        match_vals = [r[1] for r in results]
        print(f"    {mname:20s}  cos_mean={np.mean(cos_vals):.6f}  "
              f"cos_min={np.min(cos_vals):.6f}  "
              f"allclose={sum(match_vals)}/{len(match_vals)}")

    # --- Subtest 3: .T vs swap E direct comparison ---
    print("\n  --- Subtest 3: Fix A vs Fix B direct comparison ---")
    for size, label in [(10, "10×10"), (66, "66×66"), (72, "real Fock")]:
        if size == 72:
            s = samples[0]
            H = torch.tensor(s['H'] - s['H_init'], dtype=torch.float64)
            S = torch.tensor(s['S'], dtype=torch.float64)
            s_vals, s_vecs = torch.linalg.eigh(S)
            s_vals = torch.clamp(s_vals, min=1e-8)
            X = s_vecs / torch.sqrt(s_vals).unsqueeze(-2)
            nocc = s['nocc']
        else:
            torch.manual_seed(42)
            H = torch.randn(size, size); H = H @ H.T
            S = torch.eye(size); X = torch.eye(size)
            nocc = size // 3

        # Fix A
        Ha = H.clone().requires_grad_(True)
        Fsa = X.T @ Ha @ X
        Ea, Va = StableEigh_Transpose.apply(Fsa, 3.0, nocc)
        Ca = X @ Va
        la = (Ca[:, :nocc] ** 2).sum()
        ga, = torch.autograd.grad(la, Ha, retain_graph=True)

        # Fix B
        Hb = H.clone().requires_grad_(True)
        Fsb = X.T @ Hb @ X
        Eb, Vb = StableEigh_SwapE.apply(Fsb, 3.0, nocc)
        Cb = X @ Vb
        lb = (Cb[:, :nocc] ** 2).sum()
        gb, = torch.autograd.grad(lb, Hb)

        # Native
        Hn = H.clone().requires_grad_(True)
        Fsn = X.T @ Hn @ X
        En, Vn = torch.linalg.eigh(Fsn)
        Cn = X @ Vn
        ln = (Cn[:, :nocc] ** 2).sum()
        gn, = torch.autograd.grad(ln, Hn)

        c_ab = cos_sim(ga, gb)
        c_an = cos_sim(ga, gn)
        c_bn = cos_sim(gb, gn)
        match_ab = torch.allclose(ga, gb, atol=1e-6)
        match_an = torch.allclose(ga, gn, atol=1e-6)
        match_bn = torch.allclose(gb, gn, atol=1e-6)

        print(f"    {label:12s}  cos(A,B)={c_ab:.6f} match={match_ab}  "
              f"cos(A,nat)={c_an:.6f} match={match_an}  "
              f"cos(B,nat)={c_bn:.6f} match={match_bn}")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    torch.set_num_threads(1)
    
    print("=" * 65)
    print("  Verify StableEigh gradient correctness")
    print("  Device: CPU")
    print("=" * 65)

    test_a = os.environ.get("TEST_A", "1") == "1"
    test_b = os.environ.get("TEST_B", "1") == "1"
    test_c = os.environ.get("TEST_C", "1") == "1"
    test_d = os.environ.get("TEST_D", "1") == "1"
    test_e = os.environ.get("TEST_E", "1") == "1"

    if test_a:
        test_random_matrices()
    if test_b:
        test_real_fock_matrices()
    if test_c:
        test_k_matrix()
    if test_d:
        test_degenerate()
    if test_e:
        test_fix_equivalence()

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)
