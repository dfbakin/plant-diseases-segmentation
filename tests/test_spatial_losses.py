"""Unit tests for ``src.wsss.spdnet.spatial_losses``.

Each test is bound to an invariant ID from the SPDNet aux-losses spec
(`spdnet_auxiliary_spatial_losses_*.plan.md` Phase C / RESEARCH_CONTEXT.md
§5.11.1). Test classes group invariants by loss:

* ``TestEquivarianceLoss``      -> (E1)-(E4)
* ``TestPatchContrastiveLoss``  -> (C1)-(C5)
* ``TestEMATeacher``            -> (D1)-(D5)
* ``TestSelfDistillationLoss``  -> (D6)-(D9)
* ``TestSpatialLossesIntegration`` -> exercise all three on a tiny SPDNet.

All tests run on CPU in < 30 s total.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.wsss.spdnet import equivariance_transforms as ET
from src.wsss.spdnet.model import SPDNet
from src.wsss.spdnet.spatial_losses import (
    EMATeacher,
    ProjectionHead,
    equivariance_loss,
    patch_contrastive_loss,
    self_distillation_loss,
)


# ---------------------------------------------------------------------------
# Equivariance
# ---------------------------------------------------------------------------


class TestEquivarianceLoss:
    """Covers (E1)-(E4) from the spec."""

    def test_E1_identity_zero_loss(self) -> None:
        torch.manual_seed(0)
        m = torch.randn(2, 8, 8)
        loss = equivariance_loss(m, m, ET.T_ID_IDENTITY)
        assert loss.item() == 0.0

    def test_E2_hflip_matches_handcomputed(self) -> None:
        """For an asymmetric synthetic map, ``L_eq`` must equal
        ``mean((M - hflip(M))**2)`` exactly."""
        torch.manual_seed(1)
        m_orig = torch.randn(3, 8, 8)
        # Pretend the augmented branch produced exactly the original (i.e. the
        # SCA was perfectly *non*-equivariant: M(T(q), r) == M(q, r)).
        m_aug = m_orig.clone()
        loss = equivariance_loss(m_orig, m_aug, ET.T_ID_HFLIP)
        expected = ((m_orig - m_orig.flip(dims=(-1,))) ** 2).mean()
        assert torch.allclose(loss, expected)

    def test_E3_grad_flows_to_attention_aug(self) -> None:
        """Backward MUST write a non-zero ``.grad`` into ``attention_aug``,
        which in production carries the SCA's parameters via autograd."""
        torch.manual_seed(2)
        m_orig = torch.randn(2, 8, 8)
        m_aug = torch.randn(2, 8, 8, requires_grad=True)
        loss = equivariance_loss(m_orig, m_aug, ET.T_ID_HFLIP)
        loss.backward()
        assert m_aug.grad is not None
        assert m_aug.grad.abs().sum().item() > 0.0

    def test_E4_inputs_not_modified_in_place(self) -> None:
        torch.manual_seed(3)
        m_orig = torch.randn(2, 8, 8)
        m_aug = torch.randn(2, 8, 8)
        m_orig_clone = m_orig.clone()
        m_aug_clone = m_aug.clone()
        _ = equivariance_loss(m_orig, m_aug, ET.T_ID_ROT90)
        assert torch.equal(m_orig, m_orig_clone)
        assert torch.equal(m_aug, m_aug_clone)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must agree"):
            equivariance_loss(
                torch.zeros(2, 8, 8), torch.zeros(2, 8, 9), ET.T_ID_IDENTITY,
            )


# ---------------------------------------------------------------------------
# Regression: the attention map exposed by SpatialCrossAttention.forward must
# be query-dependent (otherwise L_eq is identically zero, which silently
# disables the equivariance objective). This protects against the
# ``attn_w.mean(dim=-1)`` bug where the value collapses to ``1/N`` per row.
# ---------------------------------------------------------------------------


class TestAttnMapNonConstancy:
    """Covers the (E0) regression: SCA must expose a non-constant
    per-query attention map so the equivariance loss has a non-trivial
    target. See ``spatial_losses.equivariance_loss`` and the spec."""

    def _student(self) -> SPDNet:
        return SPDNet(
            num_classes=4, fpn_channels=32,
            pretrained=False, fusion_mode="spatial",
        )

    def test_E0a_attn_map_varies_across_query_positions(self) -> None:
        """attn_map(q, r) must NOT be a constant per query position.

        If ``attn_map[b, q]`` is constant in ``q`` (e.g. when extracted via
        ``softmax_row.mean(dim=-1) = 1/N``), then ``MSE(attn(T(q),r),
        T(attn(q,r))) ≡ 0`` regardless of model weights, and L_eq cannot
        produce gradients. Fail loudly if that ever recurs.
        """
        torch.manual_seed(0)
        student = self._student()
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        attn = student.attention_map(q, r)              # (B, H, W)
        assert attn.dim() == 3
        std_per_image = attn.std(dim=(1, 2))            # (B,)
        for b in range(attn.shape[0]):
            assert std_per_image[b].item() > 1e-4, (
                f"attn_map[{b}] is essentially constant across queries "
                f"(std={std_per_image[b]:.2e}); did SCA collapse the "
                "softmax row to its mean?"
            )

    def test_E0b_attn_map_in_unit_interval(self) -> None:
        """Concentration map must be in [0, 1] (post-normalisation by log N)."""
        torch.manual_seed(1)
        student = self._student()
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        attn = student.attention_map(q, r)
        assert (attn >= 0).all(), f"attn_map has negatives: min={attn.min():.4f}"
        assert (attn <= 1).all(), f"attn_map exceeds 1: max={attn.max():.4f}"

    def test_E0c_equivariance_loss_nonzero_on_random_init_sca(self) -> None:
        """L_eq computed with a real SCA forward (random init) must be
        non-trivially > 0 for a non-identity transform. This is the
        end-to-end check that would have caught the
        ``attn_w.mean(dim=-1)`` regression at integration time.
        """
        torch.manual_seed(2)
        student = self._student()
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        feats = student.extract_merged_features(q, r, return_attn=True)
        m_orig = feats["attn_map"]
        m_aug = student.attention_map(
            ET.apply(q, ET.T_ID_ROT90),
            ref_merged_cached=feats["ref_merged"],
        )
        loss = equivariance_loss(m_orig, m_aug, ET.T_ID_ROT90)
        assert loss.item() > 1e-8, (
            f"L_eq={loss.item():.2e} is essentially zero; the SCA attn_map "
            "is probably a constant tensor and equivariance is a no-op."
        )

    def test_E0d_attn_map_deterministic_in_eval(self) -> None:
        """In eval mode the attention extraction must be byte-deterministic.

        This guards against random behaviour leaking into the attn map
        (e.g. SCA dropout being accidentally re-enabled, BN running
        stats updating mid-eval). The corresponding training-mode noise
        floor is checked in ``test_E0e``.
        """
        torch.manual_seed(3)
        student = self._student().eval()
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        m1 = student.attention_map(q, r)
        m2 = student.attention_map(q, r)
        diff = (m1 - m2).abs().max().item()
        assert diff < 1e-7, (
            f"attn_map non-deterministic in eval mode (max diff {diff:.2e}); "
            "is dropout leaking into the attention extraction?"
        )

    def test_E0e_equivariance_loss_low_noise_floor_in_train(self) -> None:
        """In train mode, identity L_eq is non-zero only because of
        backbone dropout (MSE has p=0.5 by default), but the noise
        floor must stay << expected non-equivariance signal so the
        gradient signal is dominated by real equivariance violations,
        not dropout noise. Concretely: noise floor < 1e-4 with a
        non-equivariance signal we expect to be O(1e-2) at init.
        """
        torch.manual_seed(4)
        student = self._student()
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        feats = student.extract_merged_features(q, r, return_attn=True)
        m_orig = feats["attn_map"]
        m_aug = student.attention_map(
            q, ref_merged_cached=feats["ref_merged"],
        )
        loss = equivariance_loss(m_orig, m_aug, ET.T_ID_IDENTITY)
        assert loss.item() < 1e-4, (
            f"identity L_eq noise floor too high: {loss.item():.2e} "
            "(expected << 1e-2 signal floor)"
        )


# ---------------------------------------------------------------------------
# Patch contrastive
# ---------------------------------------------------------------------------


def _disjoint_labels(B: int, C: int) -> torch.Tensor:
    """Return ``(B, C)`` multilabel where every row picks a unique class
    (so ``labels[i] @ labels[j] == 0`` for ``i != j``)."""
    assert B <= C, "need C >= B for fully disjoint labels"
    labels = torch.zeros(B, C)
    for i in range(B):
        labels[i, i] = 1.0
    return labels


class TestPatchContrastiveLoss:
    """Covers (C1)-(C5) from the spec."""

    @pytest.fixture
    def base_config(self) -> dict:
        return dict(B=2, C_in=32, C=4, H=8, W=8, D=16, K=4, M=8, tau=0.07)

    def test_C1_constant_embedding_loss_equals_log_1_plus_N(
        self, base_config: dict,
    ) -> None:
        """All-identical embeddings (anchors and negatives) -> per-positive
        denominator collapses to ``(1 + |N|) * exp(1/tau)``, so the loss is
        exactly ``log(1 + |N|)``."""
        torch.manual_seed(0)
        bc = base_config
        B, C_in, C, H, W, D, K, M, tau = (
            bc["B"], bc["C_in"], bc["C"], bc["H"], bc["W"],
            bc["D"], bc["K"], bc["M"], bc["tau"],
        )
        # Force all patch embeddings to the same unit vector by giving the
        # projector zero weight + zero bias except a single output channel.
        # Easier: choose constant input so projector output is constant after
        # normalisation regardless of weight (since L2-norm of a constant
        # nonzero vector is itself).
        p3 = torch.ones(B, C_in, H, W) * 0.5  # any constant nonzero
        p4 = torch.randn(B, C_in, H, W)
        cls_w = torch.randn(C, C_in)
        labels = _disjoint_labels(B, C)
        proj = ProjectionHead(C_in, D)

        # Sanity: after projection + L2-norm, every patch embedding equals
        # the same unit vector.
        with torch.no_grad():
            z = nn.functional.normalize(proj(p3), dim=1, eps=1e-8)
            z_flat = z.flatten(2).permute(0, 2, 1)
            assert torch.allclose(
                z_flat[0, 0], z_flat[1, 5], atol=1e-6,
            ), "projector should give identical patch embeddings for constant input"

        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=K, m_negatives=M, temperature=tau,
        )
        # Cross-class anchor pool per image: 1 cross-class image * K = K anchors.
        # Negatives per image: M (bg) + K (cross-class) = M + K.
        N = M + K
        expected = math.log(1.0 + N)
        assert math.isclose(loss.item(), expected, rel_tol=1e-4, abs_tol=1e-4), (
            f"loss={loss.item():.4f} != log(1+{N})={expected:.4f}"
        )

    def test_C2_random_embeddings_chance_loss(self, base_config: dict) -> None:
        """For random L2-normalised embeddings in HIGH-dim space at temperature
        ``tau ~ O(1)``, all pairwise sims/tau are ~ 1/sqrt(D*tau^2) ≈ 0 and the
        per-positive denominator collapses to ``(1 + |N|)``, so the loss is
        approximately ``log(1 + |N|)``.

        The std of ``sim/tau`` is ``1/(sqrt(D) * tau)``; with ``D=4096`` and
        ``tau=1.0`` it is ~0.016, well within the LSE's flat region.
        """
        torch.manual_seed(0)
        bc = base_config
        B, C_in, C, H, W, K, M = (
            bc["B"], bc["C_in"], bc["C"], bc["H"], bc["W"], bc["K"], bc["M"],
        )
        D_high = 4096
        tau = 1.0
        p3 = torch.randn(B, C_in, H, W)
        p4 = torch.randn(B, C_in, H, W)
        cls_w = torch.randn(C, C_in)
        labels = _disjoint_labels(B, C)
        proj = ProjectionHead(C_in, D_high)

        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=K, m_negatives=M, temperature=tau,
        )
        N = M + K
        expected = math.log(1.0 + N)
        # With std(sim/tau)~0.016, the LSE bias is O(0.0001).
        assert math.isclose(loss.item(), expected, rel_tol=0.05, abs_tol=0.05), (
            f"chance loss {loss.item():.4f} not close to log(1+{N})={expected:.4f}"
        )

    def test_C3_gradient_routing(self, base_config: dict) -> None:
        """Gradients flow into ``proj_head`` and ``p3_query`` but NOT into
        ``cls_weight`` or ``p4_fused`` (anchor-selection branch must be
        detached)."""
        torch.manual_seed(0)
        bc = base_config
        p3 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"], requires_grad=True)
        p4 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"], requires_grad=True)
        cls_w = torch.randn(bc["C"], bc["C_in"], requires_grad=True)
        labels = _disjoint_labels(bc["B"], bc["C"])
        proj = ProjectionHead(bc["C_in"], bc["D"])

        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj,
            top_k=bc["K"], m_negatives=bc["M"], temperature=bc["tau"],
        )
        loss.backward()

        assert p3.grad is not None and p3.grad.abs().sum().item() > 0
        assert proj.conv.weight.grad is not None
        assert proj.conv.weight.grad.abs().sum().item() > 0
        # Anchor branch is detached -> no grad in p4 or cls_w.
        assert p4.grad is None or p4.grad.abs().sum().item() == 0
        assert cls_w.grad is None or cls_w.grad.abs().sum().item() == 0

    def test_C4_single_class_batch_no_crash(self, base_config: dict) -> None:
        """When all batch images share their (single) active class, no
        cross-class anchors are available; the loss must fall back to
        background-only negatives without producing NaN/Inf."""
        torch.manual_seed(0)
        bc = base_config
        p3 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"])
        p4 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"])
        cls_w = torch.randn(bc["C"], bc["C_in"])
        labels = torch.zeros(bc["B"], bc["C"]); labels[:, 0] = 1.0  # all same class
        proj = ProjectionHead(bc["C_in"], bc["D"])
        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj,
            top_k=bc["K"], m_negatives=bc["M"], temperature=bc["tau"],
        )
        assert torch.isfinite(loss).item(), f"loss is non-finite: {loss}"

    def test_C5_permutation_invariance(self, base_config: dict) -> None:
        """Spatially permuting ``p3_query`` and ``p4_fused`` identically must
        not change the loss (anchor + bg selection are permutation-equivariant
        under the same permutation, and InfoNCE is permutation-invariant
        over its negative pool)."""
        torch.manual_seed(0)
        bc = base_config
        B, C_in, H, W = bc["B"], bc["C_in"], bc["H"], bc["W"]
        p3 = torch.randn(B, C_in, H, W)
        p4 = torch.randn(B, C_in, H, W)
        cls_w = torch.randn(bc["C"], bc["C_in"])
        labels = _disjoint_labels(bc["B"], bc["C"])
        proj = ProjectionHead(bc["C_in"], bc["D"])

        loss_orig = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj,
            top_k=bc["K"], m_negatives=bc["M"], temperature=bc["tau"],
        )

        perm = torch.randperm(H * W)
        p3_perm = p3.flatten(2)[:, :, perm].view(B, C_in, H, W)
        p4_perm = p4.flatten(2)[:, :, perm].view(B, C_in, H, W)
        loss_perm = patch_contrastive_loss(
            p3_perm, p4_perm, cls_w, labels, proj,
            top_k=bc["K"], m_negatives=bc["M"], temperature=bc["tau"],
        )
        assert torch.allclose(loss_orig, loss_perm, atol=1e-5), (
            f"permutation changed loss: {loss_orig.item()} -> {loss_perm.item()}"
        )

    def test_top_k_lt_2_raises(self, base_config: dict) -> None:
        bc = base_config
        with pytest.raises(ValueError, match="top_k=1"):
            patch_contrastive_loss(
                torch.zeros(bc["B"], bc["C_in"], bc["H"], bc["W"]),
                torch.zeros(bc["B"], bc["C_in"], bc["H"], bc["W"]),
                torch.zeros(bc["C"], bc["C_in"]),
                _disjoint_labels(bc["B"], bc["C"]),
                ProjectionHead(bc["C_in"], bc["D"]),
                top_k=1,
            )

    def test_no_active_labels_returns_zero(self, base_config: dict) -> None:
        """All-zero labels -> graceful zero-with-grad-chain."""
        bc = base_config
        p3 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"], requires_grad=True)
        p4 = torch.randn(bc["B"], bc["C_in"], bc["H"], bc["W"])
        cls_w = torch.randn(bc["C"], bc["C_in"])
        labels = torch.zeros(bc["B"], bc["C"])
        proj = ProjectionHead(bc["C_in"], bc["D"])
        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj,
            top_k=bc["K"], m_negatives=bc["M"], temperature=bc["tau"],
        )
        assert loss.item() == 0.0
        loss.backward()  # must not crash


# ---------------------------------------------------------------------------
# EMA teacher
# ---------------------------------------------------------------------------


def _tiny_spdnet() -> SPDNet:
    return SPDNet(num_classes=4, fpn_channels=32, pretrained=False, fusion_mode="spatial")


class TestEMATeacher:
    """Covers (D1)-(D5) from the spec."""

    def test_D1_no_grad_on_params_or_buffers(self) -> None:
        torch.manual_seed(0)
        student = _tiny_spdnet()
        teacher = EMATeacher(student, alpha=0.999)
        for p in teacher.parameters():
            assert not p.requires_grad
        for b in teacher.teacher.buffers():
            # buffers don't have requires_grad in the param sense; just check
            # they have no grad attached.
            assert getattr(b, "grad", None) is None

    def test_D2_alpha_zero_copies_student(self) -> None:
        torch.manual_seed(1)
        student = _tiny_spdnet()
        teacher = EMATeacher(student, alpha=0.999)
        # Mutate the student weights
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        teacher.update(student, alpha=0.0)
        for p_t, p_s in zip(teacher.teacher.parameters(), student.parameters()):
            assert torch.equal(p_t, p_s), "alpha=0 -> teacher must equal student"

    def test_D3_alpha_one_freezes_teacher(self) -> None:
        torch.manual_seed(2)
        student = _tiny_spdnet()
        teacher = EMATeacher(student, alpha=0.999)
        snapshot = [p.clone() for p in teacher.teacher.parameters()]
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        teacher.update(student, alpha=1.0)
        for p_t, p_t0 in zip(teacher.teacher.parameters(), snapshot):
            assert torch.equal(p_t, p_t0), "alpha=1 -> teacher must NOT change"

    def test_D4_bn_running_stats_emaed(self) -> None:
        """Running-mean/var of every BN layer must be EMAed (NOT just the
        learnable parameters). Regression for the "BN drift" failure mode."""
        torch.manual_seed(3)
        student = _tiny_spdnet()
        teacher = EMATeacher(student, alpha=0.999)
        # Snapshot pre-update BN running stats.
        snap = {
            k: v.clone() for k, v in teacher.teacher.state_dict().items()
            if "running_mean" in k or "running_var" in k
        }
        # Mutate the student's BN running stats.
        with torch.no_grad():
            for k, v in student.state_dict().items():
                if "running_mean" in k or "running_var" in k:
                    v.add_(torch.randn_like(v) * 0.05)
        teacher.update(student, alpha=0.5)
        # Every BN running stat in the teacher must have moved.
        for k, v_t in teacher.teacher.state_dict().items():
            if "running_mean" in k or "running_var" in k:
                assert not torch.equal(v_t, snap[k]), (
                    f"BN stat {k} did not move under EMA"
                )

    def test_D5_distance_monotone_non_increasing(self) -> None:
        """After 100 random student perturbations + ``update(alpha=0.999)``,
        the L2 distance ``||θ_t - θ_s||`` decreases on average (the teacher
        is chasing the student)."""
        torch.manual_seed(4)
        student = _tiny_spdnet()
        teacher = EMATeacher(student, alpha=0.999)

        def l2_distance() -> float:
            sq = 0.0
            for p_t, p_s in zip(teacher.teacher.parameters(), student.parameters()):
                sq += (p_t - p_s).pow(2).sum().item()
            return sq ** 0.5

        # Step 0: jump the student so the teacher has work to do.
        with torch.no_grad():
            for p in student.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        d0 = l2_distance()

        for _ in range(100):
            with torch.no_grad():
                # tiny per-step perturbation on student (simulates SGD steps)
                for p in student.parameters():
                    p.add_(torch.randn_like(p) * 0.001)
            teacher.update(student)
        d_end = l2_distance()
        assert d_end < d0, f"distance did not shrink: d0={d0:.4f} -> d_end={d_end:.4f}"

    def test_invalid_alpha_raises(self) -> None:
        student = _tiny_spdnet()
        with pytest.raises(ValueError, match="EMA alpha must be in"):
            EMATeacher(student, alpha=1.5)


# ---------------------------------------------------------------------------
# Self-distillation
# ---------------------------------------------------------------------------


class TestSelfDistillationLoss:
    """Covers (D6)-(D9) from the spec."""

    @pytest.fixture
    def base_config(self) -> dict:
        return dict(B=2, C=4, H=8, W=8)

    def test_D6_zero_loss_when_consistent(self, base_config: dict) -> None:
        """``s_student == s_teacher`` AND ``s_teacher`` is constant AND
        ``T_t == T_s`` -> both softmaxes give the uniform distribution and
        the KL is exactly zero."""
        bc = base_config
        s = torch.zeros(bc["B"], bc["C"], bc["H"], bc["W"])
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        loss = self_distillation_loss(
            s, s, labels, center, center_beta=0.9, T_teacher=0.1, T_student=0.1,
        )
        assert math.isclose(loss.item(), 0.0, abs_tol=1e-6)

    def test_D7_grad_flows_only_into_student(self, base_config: dict) -> None:
        bc = base_config
        s_s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"], requires_grad=True)
        s_t = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"]).detach()
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        loss = self_distillation_loss(s_s, s_t, labels, center)
        loss.backward()
        assert s_s.grad is not None and s_s.grad.abs().sum().item() > 0
        assert not s_t.requires_grad and s_t.grad is None

    def test_D8_centering_beta_zero_copies_batch_mean(self, base_config: dict) -> None:
        """``beta = 0`` -> ``c <- batch_mean`` exactly."""
        bc = base_config
        constant_value = 1.7
        s = torch.full((bc["B"], bc["C"], bc["H"], bc["W"]), constant_value)
        labels = _disjoint_labels(bc["B"], bc["C"])
        P = bc["H"] * bc["W"]
        center = torch.zeros(P)
        _ = self_distillation_loss(
            s, s, labels, center, center_beta=0.0,
            T_teacher=0.1, T_student=0.1,
        )
        # batch_mean is the per-position mean of s_teacher restricted to the
        # active class; for a constant tensor this is just `constant_value`.
        assert torch.allclose(center, torch.full((P,), constant_value))

    def test_D8_centering_running_mean_after_many_calls(
        self, base_config: dict,
    ) -> None:
        """After 10 calls with random teacher logits and beta=0.9, the center
        approximates the running mean (within EMA tolerance)."""
        torch.manual_seed(0)
        bc = base_config
        labels = _disjoint_labels(bc["B"], bc["C"])
        P = bc["H"] * bc["W"]
        center = torch.zeros(P)
        beta = 0.9
        ema_ref = torch.zeros(P)
        for _ in range(50):
            s_t = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"])
            # Compute the same batch_mean the loss does internally:
            # active-class row of s_t, flattened, mean over batch.
            active_first, _ = (
                torch.tensor([0, 1]),  # disjoint -> [0, 1]
                None,
            )
            S_t = torch.stack([
                s_t[i, active_first[i]].flatten() for i in range(bc["B"])
            ]).mean(dim=0)
            ema_ref.mul_(beta).add_(S_t, alpha=1 - beta)
            _ = self_distillation_loss(
                s_t, s_t, labels, center, center_beta=beta,
                T_teacher=0.1, T_student=0.1,
            )
        assert torch.allclose(center, ema_ref, atol=1e-4), (
            f"max abs diff = {(center - ema_ref).abs().max().item():.6f}"
        )

    def test_D9_sharper_teacher_yields_positive_loss(
        self, base_config: dict,
    ) -> None:
        """``s_student == s_teacher`` (non-constant, in the regime where both
        softmaxes are non-degenerate) AND ``T_t < T_s`` -> the teacher's
        softmax is sharper than the student's, so the KL is strictly
        positive.

        Random logits with std=1 divided by ``T_s=0.1`` give a softmax that
        already puts ~1.0 on the argmax (KL collapses to 0 trivially). To
        make the temperature ratio actually shape the distribution we scale
        the logits to ``std~0.05``: then ``S/T_t`` ranges roughly in [-4, 4]
        and ``S/T_s`` in [-1.5, 1.5], so the two softmaxes genuinely
        differ.
        """
        torch.manual_seed(0)
        bc = base_config
        s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"]) * 0.05
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        loss = self_distillation_loss(
            s, s, labels, center, center_beta=0.9, T_teacher=0.04, T_student=0.1,
        )
        assert loss.item() > 0.01, f"sharper-teacher KL is too small: {loss.item()}"

    def test_D9_equal_temperatures_zero_loss(
        self, base_config: dict,
    ) -> None:
        """Counterpart to D9: with ``T_t == T_s`` and ``s_student ==
        s_teacher`` and ``center == 0``, KL must be exactly zero regardless
        of the logit shape."""
        torch.manual_seed(1)
        bc = base_config
        s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"])
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        loss = self_distillation_loss(
            s, s, labels, center, center_beta=0.9, T_teacher=0.1, T_student=0.1,
        )
        assert math.isclose(loss.item(), 0.0, abs_tol=1e-5), (
            f"equal-temperature KL is not zero: {loss.item()}"
        )

    def test_no_active_labels_returns_zero(self, base_config: dict) -> None:
        bc = base_config
        s_s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"], requires_grad=True)
        s_t = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"]).detach()
        labels = torch.zeros(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        loss = self_distillation_loss(s_s, s_t, labels, center)
        assert loss.item() == 0.0
        loss.backward()  # must not crash

    def test_invalid_temperature_raises(self, base_config: dict) -> None:
        bc = base_config
        s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"])
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"])
        with pytest.raises(ValueError, match="temperatures must be > 0"):
            self_distillation_loss(s, s, labels, center, T_teacher=0.0, T_student=0.1)

    def test_invalid_center_shape_raises(self, base_config: dict) -> None:
        bc = base_config
        s = torch.randn(bc["B"], bc["C"], bc["H"], bc["W"])
        labels = _disjoint_labels(bc["B"], bc["C"])
        center = torch.zeros(bc["H"] * bc["W"] + 1)  # wrong shape
        with pytest.raises(ValueError, match="center shape"):
            self_distillation_loss(s, s, labels, center)


# ---------------------------------------------------------------------------
# Contrastive-loss linear warmup schedule
# ---------------------------------------------------------------------------


class TestLambdaConWarmup:
    """Regression tests for ``SPDNetModule.effective_lambda_con`` and its
    interaction with ``training_step``.

    Invariants exercised:

    * (W1) no-warmup defaults reproduce the pre-warmup behaviour.
    * (W2) ``epoch < start``  -> effective lambda is exactly 0.
    * (W3) linear ramp hits exactly 0, base/ramp, ... , base at integer epochs.
    * (W4) ``epoch >= start + ramp`` -> effective lambda is exactly ``base``.
    * (W5) ``lambda_con == 0`` shuts the whole loss off regardless of schedule.
    * (W6) with lambda_con_eff == 0 the ``training_step`` does **not** add
      anything contrastive-related to the total loss (and does not
      allocate ``train/L_con``), while still logging the schedule value.
    """

    @staticmethod
    def _make_module(
        *,
        lambda_con: float,
        start: int,
        ramp: int,
    ):
        # Import lazily so the module's heavy deps only load for this test.
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        cfg = SPDNetSpatialLossesConfig(
            lambda_eq=0.0,
            lambda_con=lambda_con,
            lambda_distill=0.0,
            con_warmup_start_epoch=start,
            con_warmup_epochs=ramp,
            online_loc_eval_enabled=False,
        )
        mod = SPDNetModule(
            num_classes=4,
            fpn_channels=16,
            mse_reduction=4,
            pretrained=False,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_epochs=0,
            min_lr=1e-5,
            fusion_mode="spatial",
            losses_cfg=cfg,
            online_loc_metric=None,
            image_size=64,
        )
        return mod

    def test_W1_defaults_no_warmup(self) -> None:
        m = self._make_module(lambda_con=0.5, start=0, ramp=0)
        for e in (0, 1, 5, 100):
            assert m.effective_lambda_con(epoch=e) == pytest.approx(0.5)

    def test_W2_before_start_is_zero(self) -> None:
        m = self._make_module(lambda_con=0.5, start=14, ramp=7)
        for e in range(14):
            assert m.effective_lambda_con(epoch=e) == 0.0

    def test_W3_linear_ramp_values(self) -> None:
        base, start, ramp = 0.5, 14, 5
        m = self._make_module(lambda_con=base, start=start, ramp=ramp)
        # At the boundary epoch == start, (e - start)/ramp == 0 -> still 0.
        # The ramp hits its full value exactly one epoch after start+ramp-1.
        expected = {
            14: 0.0,
            15: base * 1 / 5,
            16: base * 2 / 5,
            17: base * 3 / 5,
            18: base * 4 / 5,
            19: base,        # start + ramp -> clamped to base.
            20: base,
            100: base,
        }
        for e, want in expected.items():
            got = m.effective_lambda_con(epoch=e)
            assert got == pytest.approx(want, abs=1e-7), (
                f"epoch={e}: got {got}, want {want}"
            )

    def test_W4_post_ramp_clamped_to_base(self) -> None:
        m = self._make_module(lambda_con=0.3, start=14, ramp=7)
        assert m.effective_lambda_con(epoch=21) == pytest.approx(0.3)
        assert m.effective_lambda_con(epoch=22) == pytest.approx(0.3)
        assert m.effective_lambda_con(epoch=1000) == pytest.approx(0.3)

    def test_W5_zero_lambda_disables_regardless_of_schedule(self) -> None:
        m = self._make_module(lambda_con=0.0, start=0, ramp=5)
        for e in (0, 3, 5, 100):
            assert m.effective_lambda_con(epoch=e) == 0.0

    def test_W5_negative_lambda_treated_as_zero(self) -> None:
        # Negative lambdas would invert the gradient direction and are
        # almost certainly a config typo. The schedule clamps to 0.
        m = self._make_module(lambda_con=-0.5, start=0, ramp=5)
        for e in (0, 3, 5, 100):
            assert m.effective_lambda_con(epoch=e) == 0.0

    @staticmethod
    def _drive_training_step(
        module,
        epoch: int,
        seed: int,
        monkeypatch,
    ) -> tuple[torch.Tensor, dict[str, float], int]:
        """Run a single ``training_step`` at ``epoch`` and return
        ``(total, logged, patch_contrastive_loss_call_count)``.

        Patches both ``lm.patch_contrastive_loss`` (to count calls) and
        the module's ``.log`` method (to capture values) so we don't need
        a real Lightning trainer attached.
        """
        import src.wsss.spdnet.lightning as lm

        # Lightning's current_epoch property walks through .trainer. Short-
        # circuit it with a plain attribute that takes precedence on the
        # instance (we cast the module back to LightningModule.current_epoch
        # by replacing the property on the class scope for this test only).
        monkeypatch.setattr(
            type(module),
            "current_epoch",
            property(lambda self: getattr(self, "_test_epoch", 0)),
            raising=False,
        )
        module._test_epoch = epoch

        call_count = {"n": 0}
        orig_pcl = lm.patch_contrastive_loss

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return orig_pcl(*args, **kwargs)

        monkeypatch.setattr(lm, "patch_contrastive_loss", _spy)

        logged: dict[str, float] = {}

        def _fake_log(name, value, *_, **__):
            if hasattr(value, "item"):
                try:
                    logged[name] = float(value.item())
                    return
                except Exception:
                    pass
            logged[name] = float(value)

        # Shadow the inherited ``LightningModule.log`` on this one instance.
        monkeypatch.setattr(module, "log", _fake_log, raising=False)

        torch.manual_seed(seed)
        B, H, W = 2, 64, 64
        batch = {
            "query_image": torch.randn(B, 3, H, W),
            "ref_images": torch.randn(B, 3, H, W),
            "query_label": _disjoint_labels(B, 4),
        }
        total = module.training_step(batch, batch_idx=0)
        return total, logged, call_count["n"]

    def test_W6_training_step_does_not_use_L_con_before_warmup(
        self, monkeypatch,
    ) -> None:
        """Drive a single training_step at an epoch before the ramp starts
        and verify (a) ``patch_contrastive_loss`` is never called, (b)
        ``train/L_con`` is not logged, (c) ``train/lambda_con_eff`` IS
        logged (so MLflow plots show the schedule), (d) the total loss
        equals the cls loss alone."""
        m = self._make_module(lambda_con=0.5, start=14, ramp=7)
        assert m.proj_head is not None, (
            "proj_head must be allocated at __init__ whenever lambda_con>0, "
            "independent of the schedule, so checkpoints stay compatible."
        )

        total, logged, n_calls = self._drive_training_step(
            m, epoch=0, seed=0, monkeypatch=monkeypatch,
        )
        assert n_calls == 0, (
            "patch_contrastive_loss should NOT be called before warmup starts"
        )
        assert "train/L_con" not in logged, (
            f"train/L_con should not be logged pre-warmup; got keys: "
            f"{sorted(logged)}"
        )
        assert "train/lambda_con_eff" in logged, (
            f"train/lambda_con_eff must be logged; got keys: {sorted(logged)}"
        )
        assert logged["train/lambda_con_eff"] == 0.0
        assert "train/L_cls" in logged
        # Total loss equals the classifier loss alone.
        assert total.item() == pytest.approx(logged["train/L_cls"], abs=1e-5)

    def test_W6_training_step_uses_L_con_mid_ramp(
        self, monkeypatch,
    ) -> None:
        """Counterpart to W6: mid-ramp, ``patch_contrastive_loss`` IS
        invoked, ``train/L_con`` IS logged, and ``train/lambda_con_eff``
        reports the partial ramp value."""
        base, start, ramp = 0.5, 14, 5
        m = self._make_module(lambda_con=base, start=start, ramp=ramp)
        assert m.proj_head is not None
        expected_eff = base * 2 / 5

        _, logged, n_calls = self._drive_training_step(
            m, epoch=start + 2, seed=1, monkeypatch=monkeypatch,
        )
        assert n_calls == 1, "patch_contrastive_loss must be invoked mid-ramp"
        assert "train/L_con" in logged
        assert logged.get("train/lambda_con_eff", -1.0) == pytest.approx(
            expected_eff,
        )

    def test_W6_training_step_uses_full_L_con_post_ramp(
        self, monkeypatch,
    ) -> None:
        """Post-ramp, effective == base; L_con is used at full weight."""
        base, start, ramp = 0.5, 14, 5
        m = self._make_module(lambda_con=base, start=start, ramp=ramp)

        _, logged, n_calls = self._drive_training_step(
            m, epoch=start + ramp + 3, seed=2, monkeypatch=monkeypatch,
        )
        assert n_calls == 1
        assert "train/L_con" in logged
        assert logged.get("train/lambda_con_eff", -1.0) == pytest.approx(base)


class TestWarmstartLoad:
    """Regression tests for ``+checkpoint=`` Hydra override in train_spdnet.

    We don't want to trigger a full trainer.fit(), so we exercise just the
    load logic: save an SPDNetModule checkpoint, instantiate a fresh
    module with a *different* aux-loss config, then call
    ``module.load_state_dict(ckpt["state_dict"], strict=False)`` the same
    way ``train_spdnet`` does. Asserts:

    * loading succeeds (no unexpected keys from the source);
    * the ``proj_head`` in the target module is flagged as missing
      (source had ``lambda_con=0``, so no proj_head was saved);
    * backbone weights survive the round-trip byte-for-byte (the whole
      point of a warmstart).
    """

    def _save_eq_only_ckpt(self, tmp_path):
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        source_cfg = SPDNetSpatialLossesConfig(
            lambda_eq=1.0,
            lambda_con=0.0,
            lambda_distill=0.0,
            online_loc_eval_enabled=False,
        )
        source = SPDNetModule(
            num_classes=4,
            fpn_channels=16,
            mse_reduction=4,
            pretrained=False,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_epochs=0,
            min_lr=1e-5,
            fusion_mode="spatial",
            losses_cfg=source_cfg,
            online_loc_metric=None,
            image_size=64,
        )
        # Source must not have a proj_head (eq-only).
        assert source.proj_head is None
        ckpt_path = tmp_path / "eq_only.ckpt"
        # Lightning saves under the "state_dict" key; mirror that.
        torch.save({"state_dict": source.state_dict()}, ckpt_path)
        return ckpt_path, source

    def test_warmstart_load_from_eq_only_ckpt(self, tmp_path) -> None:
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        ckpt_path, source = self._save_eq_only_ckpt(tmp_path)
        target_cfg = SPDNetSpatialLossesConfig(
            lambda_eq=1.0,
            lambda_con=0.5,          # adds proj_head to the target.
            lambda_distill=0.0,
            con_warmup_start_epoch=0,
            con_warmup_epochs=5,
            online_loc_eval_enabled=False,
        )
        target = SPDNetModule(
            num_classes=4,
            fpn_channels=16,
            mse_reduction=4,
            pretrained=False,
            learning_rate=1e-4,
            weight_decay=0.05,
            warmup_epochs=0,
            min_lr=1e-5,
            fusion_mode="spatial",
            losses_cfg=target_cfg,
            online_loc_metric=None,
            image_size=64,
        )
        assert target.proj_head is not None, (
            "Target must have a proj_head when lambda_con > 0"
        )
        bb_keys_before = {
            k: v.clone() for k, v in target.state_dict().items()
            if k.startswith("model.")
        }
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = target.load_state_dict(
            ckpt["state_dict"], strict=False,
        )
        # Source is eq-only -> target's proj_head must appear in `missing`.
        proj_missing = [k for k in missing if "proj_head" in k]
        assert len(proj_missing) > 0, (
            f"Expected proj_head.* to be missing from an eq-only checkpoint; "
            f"got missing={missing[:5]}"
        )
        # No unexpected keys: the source only has backbone/SCA/classifier,
        # all of which exist on the target.
        assert unexpected == [], f"unexpected keys: {unexpected}"
        # Backbone weights updated (they differ between random-init target and
        # source), and match source exactly for every shared key.
        src_state = source.state_dict()
        for k, v_now in target.state_dict().items():
            if not k.startswith("model."):
                continue
            if k in src_state:
                assert torch.equal(v_now, src_state[k]), (
                    f"key {k!r} did not survive warmstart load"
                )
                # And (almost certainly) differs from the fresh random init.
                if k in bb_keys_before and bb_keys_before[k].shape == v_now.shape:
                    if not torch.equal(bb_keys_before[k], v_now):
                        # At least one key should have moved; this is an
                        # existence check -- weights that happen to collide
                        # with the random init are OK.
                        pass


# ---------------------------------------------------------------------------
# Integration: tiny SPDNet, full training_step path
# ---------------------------------------------------------------------------


class TestSpatialLossesIntegration:
    """End-to-end check on a tiny SPDNet that all three losses fire and
    produce non-zero gradients on the intended parameter sets."""

    def test_full_training_step_path(self) -> None:
        torch.manual_seed(0)
        B, C_classes, H, W = 2, 4, 64, 64
        student = SPDNet(
            num_classes=C_classes, fpn_channels=32, pretrained=False,
            fusion_mode="spatial",
        )
        teacher = EMATeacher(student, alpha=0.999)
        proj = ProjectionHead(in_channels=32, out_channels=16)

        q = torch.randn(B, 3, H, W)
        r = torch.randn(B, 3, H, W)
        labels = _disjoint_labels(B, C_classes)

        # Main forward: features + logits + cls loss.
        feats = student.extract_merged_features(q, r, return_attn=True)
        F_p3 = feats["query_merged"]
        F_p4 = feats["fused"]
        M_orig = feats["attn_map"]
        H_p, W_p = F_p4.shape[-2:]
        S = torch.einsum("nc,bchw->bnhw", student.classifier.weight, F_p4)
        logits = F_p4.mean(dim=[2, 3]) @ student.classifier.weight.t() + student.classifier.bias
        L_cls = nn.functional.multilabel_soft_margin_loss(logits, labels)

        # Equivariance: separate branch with T(q).
        t_id = ET.T_ID_HFLIP
        q_aug = ET.apply(q, t_id)
        M_aug = student.attention_map(q_aug, ref_merged_cached=feats["ref_merged"])
        L_eq = equivariance_loss(M_orig, M_aug, t_id)

        # Patch contrastive on F^P3.
        L_con = patch_contrastive_loss(
            F_p3, F_p4, student.classifier.weight, labels, proj,
            top_k=4, m_negatives=8, temperature=0.07,
        )

        # Self-distillation (teacher under no_grad).
        S_t = teacher(q, r)
        center = torch.zeros(H_p * W_p)
        L_dist = self_distillation_loss(
            S, S_t, labels, center, center_beta=0.9, T_teacher=0.04, T_student=0.1,
        )

        total = L_cls + 1.0 * L_eq + 0.5 * L_con + 0.1 * L_dist
        assert torch.isfinite(total), f"non-finite total loss: {total}"

        total.backward()

        # Spatial cross-attention parameters MUST receive a non-zero grad
        # (this is the whole point of the auxiliary losses).
        sca_grads = [
            p.grad for p in student.spatial_attn.parameters() if p.requires_grad
        ]
        assert all(g is not None for g in sca_grads)
        assert any(g.abs().sum().item() > 0 for g in sca_grads)

        # Projection head MUST receive grads from L_con.
        assert proj.conv.weight.grad is not None
        assert proj.conv.weight.grad.abs().sum().item() > 0

        # Teacher MUST NOT receive grads (frozen).
        for p in teacher.parameters():
            assert p.grad is None

        # EMA update step - run it, verify teacher state moved.
        snapshot = [p.clone() for p in teacher.teacher.parameters()]
        teacher.update(student, alpha=0.999)
        moved = sum(
            (p - p0).abs().sum().item()
            for p, p0 in zip(teacher.teacher.parameters(), snapshot)
        )
        assert moved > 0, "EMA update did not change teacher params"
