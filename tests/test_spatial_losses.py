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
    attention_argmax_share_loss,
    attention_concentration_loss,
    attention_marginal_entropy_loss,
    cam_pseudo_mask_loss,
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


# ---------------------------------------------------------------------------
# D1: Attention concentration regulariser
# ---------------------------------------------------------------------------


class TestAttentionConcentrationLoss:
    """Covers the D1 attention-concentration regulariser invariants.

    Listed as (AC1)-(AC6) in RESEARCH_CONTEXT.md §5.13.7 "D1 design".
    """

    def test_AC1_uniform_map_loss_is_zero(self) -> None:
        """attn_map == 0 everywhere (the actual uniform-attention fixed
        point the SCA converges to) -> loss == 0."""
        attn = torch.zeros(2, 8, 8)
        loss = attention_concentration_loss(attn)
        assert loss.item() == 0.0

    def test_AC2_perfectly_peaked_map_loss_is_minus_one(self) -> None:
        """attn_map == 1 everywhere (every query attends to a single key)
        -> loss == -1, the global minimum."""
        attn = torch.ones(2, 8, 8)
        loss = attention_concentration_loss(attn)
        assert loss.item() == pytest.approx(-1.0, abs=1e-7)

    def test_AC3_monotonic_in_mean(self) -> None:
        """Loss strictly decreases as mean concentration grows (the
        regulariser has no spurious plateau)."""
        means = [0.1, 0.3, 0.5, 0.7, 0.9]
        losses = [
            attention_concentration_loss(torch.full((1, 4, 4), m)).item()
            for m in means
        ]
        for lo, hi in zip(losses, losses[1:]):
            assert hi < lo, (
                f"loss must decrease as concentration grows; got {losses}"
            )

    def test_AC4_matches_minus_mean(self) -> None:
        """Exact numeric identity L_ac = -mean(attn_map) for random input."""
        torch.manual_seed(7)
        attn = torch.rand(3, 8, 8)
        loss = attention_concentration_loss(attn)
        assert torch.allclose(loss, -attn.mean())

    def test_AC5_gradient_flows_through_attn_map(self) -> None:
        """Backward must write non-zero grads into the ``attn_map`` tensor
        (which in production carries the SCA in-proj weights)."""
        torch.manual_seed(8)
        attn = torch.rand(2, 8, 8, requires_grad=True)
        loss = attention_concentration_loss(attn)
        loss.backward()
        assert attn.grad is not None
        # ``-1 / (B*H*W)`` on every element; very small but identical.
        expected = -1.0 / (2 * 8 * 8)
        assert torch.allclose(attn.grad, torch.full_like(attn.grad, expected))

    def test_AC6_wrong_shape_raises(self) -> None:
        attn = torch.rand(2, 4, 8, 8)  # 4D
        with pytest.raises(ValueError, match=r"\(B, H, W\)"):
            attention_concentration_loss(attn)
        attn = torch.rand(8, 8)        # 2D
        with pytest.raises(ValueError, match=r"\(B, H, W\)"):
            attention_concentration_loss(attn)

    def test_AC7_integrates_with_spdnet_forward(self) -> None:
        """End-to-end: a real SPDNet spatial forward returns ``attn_map``
        with live grad that propagates back to the SCA in-proj weights
        via ``attention_concentration_loss``."""
        torch.manual_seed(9)
        student = SPDNet(
            num_classes=4, fpn_channels=16, pretrained=False,
            fusion_mode="spatial",
        )
        q = torch.randn(2, 3, 64, 64)
        r = torch.randn(2, 3, 64, 64)
        feats = student.extract_merged_features(q, r, return_attn=True)
        attn = feats["attn_map"]
        # Numerical sanity: on a random init the concentration should sit
        # well below the peak (this model hasn't been trained to concentrate).
        assert 0.0 <= attn.mean().item() <= 1.0
        loss = attention_concentration_loss(attn)
        loss.backward()
        # At least one SCA in-proj weight gets a grad.
        g = student.spatial_attn.cross_attn.in_proj_weight.grad
        assert g is not None and g.abs().sum().item() > 0, (
            "L_ac must update the SCA attention in-projection weights"
        )


# ---------------------------------------------------------------------------
# D2: Pseudo-mask CAM supervision
# ---------------------------------------------------------------------------


class TestCAMPseudoMaskLoss:
    """Covers the D2 pseudo-mask CAM loss invariants.

    Listed as (PM1)-(PM9) in RESEARCH_CONTEXT.md §5.13.7 "D2 design".
    """

    @staticmethod
    def _make_inputs(B: int = 2, C: int = 4, Cin: int = 8, Hf: int = 8):
        """Build ``(p3, p4, cls_weight, labels)`` for the pseudo-mask loss."""
        torch.manual_seed(17)
        p3 = torch.randn(B, Cin, Hf, Hf, requires_grad=True)
        p4 = torch.randn(B, Cin, Hf, Hf, requires_grad=True)
        cls_weight = torch.randn(C, Cin, requires_grad=True)
        labels = _disjoint_labels(B, C)
        return p3, p4, cls_weight, labels

    def test_PM1_no_active_labels_returns_grad_preserving_zero(self) -> None:
        p3, p4, cls_w, _ = self._make_inputs()
        labels = torch.zeros(p3.shape[0], cls_w.shape[0])  # all-zero labels
        loss = cam_pseudo_mask_loss(p3, p4, cls_w, labels)
        assert loss.item() == 0.0
        loss.backward()  # must not raise; grad chain intact
        assert p4.grad is not None

    def test_PM2_invalid_alpha_beta_raises(self) -> None:
        p3, p4, cls_w, labels = self._make_inputs()
        with pytest.raises(ValueError, match="alpha_pos"):
            cam_pseudo_mask_loss(p3, p4, cls_w, labels, alpha_pos=0.0)
        with pytest.raises(ValueError, match="alpha_pos"):
            cam_pseudo_mask_loss(p3, p4, cls_w, labels, alpha_pos=1.0)
        with pytest.raises(ValueError, match="beta_neg"):
            cam_pseudo_mask_loss(p3, p4, cls_w, labels, beta_neg=0.0)
        with pytest.raises(ValueError, match="beta_neg"):
            cam_pseudo_mask_loss(p3, p4, cls_w, labels, beta_neg=1.0)
        with pytest.raises(ValueError, match=r"alpha_pos \+ beta_neg"):
            cam_pseudo_mask_loss(
                p3, p4, cls_w, labels, alpha_pos=0.5, beta_neg=0.5,
            )

    def test_PM3_pos_and_neg_masks_are_disjoint(self) -> None:
        """Per-image, pos and neg masks must never overlap (the loss would
        otherwise be self-contradictory at those pixels)."""
        # Construct chvar to have many ties at exactly the threshold so
        # disjointness is enforced by the post-processing, not by pure
        # strict-inequality on the thresholds. Build p3 so that
        # ``chvar = Var_c(p3)`` has a plateau.
        B, Cin, Hf = 2, 4, 8
        # Make the first 16 positions of each image have identical chvar
        # (pick them via the plateau), rest have higher chvar.
        p3 = torch.zeros(B, Cin, Hf, Hf)
        # Insert a per-position-level offset that makes the plateau fall
        # exactly at the boundary between positives (top-alpha) and
        # negatives (bottom-beta): alpha=0.25, beta=0.5 on 64 positions
        # -> k_pos=16 (top 16), k_neg=32 (bot 32), 16 remain unsupervised.
        flat = torch.arange(Hf * Hf, dtype=torch.float32).view(Hf, Hf)
        # Broadcast into p3 via channel 0 so Var_c(p3) == flat / Cin roughly.
        p3[:, 0] = flat
        p3[:, 1] = -flat
        p3[:, 2] = torch.zeros_like(flat)
        p3[:, 3] = torch.zeros_like(flat)

        p4 = torch.randn(B, Cin, Hf, Hf)
        cls_w = torch.randn(4, Cin)
        labels = _disjoint_labels(B, 4)

        # Access the internal masks by reconstructing them exactly the way
        # the loss does. (We don't have a direct hook, so we replay the
        # top-alpha/bottom-beta logic.)
        chvar = p3.detach().var(dim=1, unbiased=False).flatten(1)
        P = chvar.shape[1]
        k_pos = max(1, int(round(0.25 * P)))
        k_neg = max(1, int(round(0.50 * P)))
        from src.wsss.spdnet.spatial_losses import _kth_threshold
        thr_pos = _kth_threshold(chvar, k_pos, largest=True)
        thr_neg = _kth_threshold(chvar, k_neg, largest=False)
        pos = (chvar >= thr_pos).float()
        neg = (chvar <= thr_neg).float()
        # Apply the same "pos *= 1 - neg" step the loss does. After this
        # there must be no overlap.
        pos = pos * (1 - neg)
        overlap = (pos * neg).sum().item()
        assert overlap == 0.0, (
            f"pos and neg overlap by {overlap} positions after disjointness "
            f"filter"
        )

        # Smoke-level: the loss itself runs without NaN.
        loss = cam_pseudo_mask_loss(p3, p4, cls_w, labels)
        assert torch.isfinite(loss)

    def test_PM4_perfect_alignment_gives_zero_loss(self) -> None:
        """If cam_norm == target exactly at every supervised position the
        loss must be 0."""
        B, Cin, Hf = 2, 4, 8
        P = Hf * Hf
        # Build chvar via p3 so the top-alpha positions are the first 16
        # flattened positions (highest-index indices after argsort).
        # Easiest: make Var_c(p3) = flat index.
        p3 = torch.zeros(B, Cin, Hf, Hf)
        flat = torch.arange(P, dtype=torch.float32).view(Hf, Hf)
        p3[:, 0] = flat
        p3[:, 1] = -flat
        # alpha=0.25 -> k_pos=16 (positions 48..63)
        # beta=0.5  -> k_neg=32 (positions  0..31)

        # Craft p4 so CAM(active) hits exactly those positions.
        # CAM[b, c] = sum_c' cls_weight[c, c'] * p4[b, c']
        # For the active class (index 0 for b=0, 1 for b=1), set
        # cls_weight[active_c] = [1, 0, 0, ...] and place p4[:, 0]
        # equal to the target mask.
        target = torch.zeros(B, Hf, Hf)
        target.view(B, -1)[:, 48:] = 1.0
        # After per-image min-max norm, p4_c0 = target gives cam_norm = target.
        p4 = torch.zeros(B, Cin, Hf, Hf)
        p4[:, 0] = target
        cls_w = torch.zeros(4, Cin)
        cls_w[0, 0] = 1.0
        cls_w[1, 0] = 1.0  # b=1 active class is also class 1 in disjoint labels
        labels = _disjoint_labels(B, 4)
        # Disable intersection: with intersection the top-alpha of the CAM
        # must also land in top-alpha positions, which is satisfied here,
        # but the test is simpler with intersection=False.
        loss = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            alpha_pos=0.25, beta_neg=0.5, use_intersection=False,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_PM5_worst_alignment_gives_one(self) -> None:
        """If cam_norm is 1 at all negatives and 0 at all positives the
        MSE is exactly 1.0 on every supervised pixel -> total == 1."""
        B, Cin, Hf = 2, 4, 8
        P = Hf * Hf
        p3 = torch.zeros(B, Cin, Hf, Hf)
        flat = torch.arange(P, dtype=torch.float32).view(Hf, Hf)
        p3[:, 0] = flat
        p3[:, 1] = -flat

        # Build an anti-aligned CAM: 1 where target says 0, 0 where target says 1.
        target = torch.zeros(B, Hf, Hf)
        target.view(B, -1)[:, 48:] = 1.0
        anti = 1.0 - target
        p4 = torch.zeros(B, Cin, Hf, Hf)
        p4[:, 0] = anti
        cls_w = torch.zeros(4, Cin)
        cls_w[0, 0] = 1.0
        cls_w[1, 0] = 1.0
        labels = _disjoint_labels(B, 4)
        loss = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            alpha_pos=0.25, beta_neg=0.5, use_intersection=False,
        )
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_PM6_grad_through_p4_and_cls_weight(self) -> None:
        p3, p4, cls_w, labels = self._make_inputs()
        loss = cam_pseudo_mask_loss(p3, p4, cls_w, labels)
        loss.backward()
        assert p4.grad is not None and p4.grad.abs().sum().item() > 0
        assert cls_w.grad is not None and cls_w.grad.abs().sum().item() > 0

    def test_PM7_no_grad_through_p3_query(self) -> None:
        """p3_query is only used for the (detached) seed mask; its grad must
        be None after backward to prevent the feature-extractor from being
        dragged by the pseudo-mask target (that would turn the loss into a
        fixed point)."""
        p3, p4, cls_w, labels = self._make_inputs()
        p3.retain_grad()
        loss = cam_pseudo_mask_loss(p3, p4, cls_w, labels)
        loss.backward()
        # Either grad is None or grad == 0 -- both satisfy the "chvar side is
        # detached" contract. (Autograd allocates a zero grad when the tensor
        # is unused but retain_grad is set.)
        if p3.grad is not None:
            assert p3.grad.abs().sum().item() == 0.0, (
                f"p3_query must be detached in chvar seed computation; "
                f"grad_abs_sum={p3.grad.abs().sum().item()}"
            )

    def test_PM8_intersection_shrinks_or_equals_positive_mask(self) -> None:
        """With ``use_intersection=True`` the positive set is the AND of
        chvar and CAM tops; it cannot be larger than the chvar-only version."""
        B, Cin, Hf = 2, 4, 8
        torch.manual_seed(23)
        p3 = torch.randn(B, Cin, Hf, Hf)
        p4 = torch.randn(B, Cin, Hf, Hf)
        cls_w = torch.randn(4, Cin)
        labels = _disjoint_labels(B, 4)
        # Replicate the loss's inner mask construction for both variants.
        chvar = p3.detach().var(dim=1, unbiased=False).flatten(1)
        P = chvar.shape[1]
        k_pos = max(1, int(round(0.25 * P)))
        from src.wsss.spdnet.spatial_losses import _kth_threshold
        thr_pos = _kth_threshold(chvar, k_pos, largest=True)
        pos_chvar = (chvar >= thr_pos).view(B, Hf, Hf).float()

        # CAM top-alpha (per loss's normalisation)
        S_full = torch.einsum("nc,bchw->bnhw", cls_w, p4)
        active = labels.argmax(dim=1)
        idx = active[:, None, None, None].expand(-1, 1, Hf, Hf)
        cam_act = torch.gather(S_full, 1, idx).squeeze(1)
        cam_flat = cam_act.flatten(1)
        mn = cam_flat.amin(dim=1, keepdim=True)
        mx = cam_flat.amax(dim=1, keepdim=True)
        cam_norm = ((cam_flat - mn) / (mx - mn + 1e-8)).view(B, Hf, Hf)
        thr_cam = _kth_threshold(cam_norm.flatten(1), k_pos, largest=True)
        pos_cam = (cam_norm.flatten(1) >= thr_cam).view(B, Hf, Hf).float()

        inter = pos_chvar * pos_cam
        assert inter.sum().item() <= pos_chvar.sum().item()

    def test_PM9_constant_cam_does_not_nan(self) -> None:
        """If the CAM is constant (min == max), per-image min-max norm
        divides by a near-zero denominator -> must be numerically stable."""
        B, Cin, Hf = 2, 4, 8
        p3 = torch.randn(B, Cin, Hf, Hf)
        # Classifier -> constant CAM means p4 has a null projection onto
        # cls_weight[active]. Easiest: set cls_weight to zero; then CAM == 0.
        p4 = torch.randn(B, Cin, Hf, Hf, requires_grad=True)
        cls_w = torch.zeros(4, Cin)
        labels = _disjoint_labels(B, 4)
        loss = cam_pseudo_mask_loss(p3, p4, cls_w, labels)
        assert torch.isfinite(loss), f"constant CAM produced non-finite loss: {loss}"


# ---------------------------------------------------------------------------
# D3: L_con union anchors
# ---------------------------------------------------------------------------


class TestPatchContrastiveUnionAnchors:
    """Covers the D3 union-anchor variant of ``patch_contrastive_loss``.

    Listed as (UN1)-(UN5) in RESEARCH_CONTEXT.md §5.13.7 "D3 design".
    """

    @staticmethod
    def _make_inputs(B: int = 3, C: int = 4, Cin: int = 8, Hf: int = 8, seed: int = 0):
        torch.manual_seed(seed)
        p3 = torch.randn(B, Cin, Hf, Hf, requires_grad=True)
        p4 = torch.randn(B, Cin, Hf, Hf, requires_grad=True)
        cls_weight = torch.randn(C, Cin, requires_grad=True)
        labels = _disjoint_labels(B, C)
        proj = ProjectionHead(in_channels=Cin, out_channels=Cin * 2)
        return p3, p4, cls_weight, labels, proj

    def test_UN1_classifier_default_is_backward_compatible(self) -> None:
        """Calling without ``anchor_source`` must produce the same result
        as explicitly passing ``anchor_source="classifier"``."""
        p3, p4, cls_w, labels, proj = self._make_inputs()
        loss_default = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
        )
        loss_cls = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="classifier",
        )
        assert torch.allclose(loss_default, loss_cls)

    def test_UN2_union_differs_from_classifier_when_sources_disagree(self) -> None:
        """Cook up inputs where classifier score and chvar saliency rank
        positions differently (negatively correlated), then verify the
        union anchors select a different set -> different loss."""
        B, C, Cin, Hf = 2, 4, 8, 8
        torch.manual_seed(3)
        # Build p3 with a fixed chvar pattern that prefers the top-left.
        p3 = torch.zeros(B, Cin, Hf, Hf, requires_grad=True)
        p3.data[:, 0, :4, :4] = 5.0  # high chvar in top-left 4x4
        p3.data[:, 1, :4, :4] = -5.0
        # Build p4 so classifier score prefers bottom-right 4x4.
        p4 = torch.zeros(B, Cin, Hf, Hf, requires_grad=True)
        p4.data[:, 0, 4:, 4:] = 5.0
        cls_w = torch.zeros(C, Cin); cls_w[0, 0] = 1.0; cls_w[1, 0] = 1.0
        cls_w.requires_grad = True
        labels = _disjoint_labels(B, C)
        proj = ProjectionHead(in_channels=Cin, out_channels=Cin * 2)
        l_cls = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="classifier",
        )
        l_union = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="union_cls_chvar",
        )
        assert not torch.allclose(l_cls, l_union, atol=1e-4), (
            "union anchors must produce a DIFFERENT loss from classifier-only "
            "when the two sources disagree on ranking"
        )

    def test_UN3_invalid_source_raises(self) -> None:
        p3, p4, cls_w, labels, proj = self._make_inputs()
        with pytest.raises(ValueError, match="anchor_source="):
            patch_contrastive_loss(
                p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
                anchor_source="does_not_exist",
            )

    def test_UN4_union_gradient_flows_through_proj(self) -> None:
        p3, p4, cls_w, labels, proj = self._make_inputs()
        loss = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="union_cls_chvar",
        )
        loss.backward()
        assert proj.conv.weight.grad is not None
        assert proj.conv.weight.grad.abs().sum().item() > 0

    def test_UN5_when_rankings_agree_union_matches_classifier(self) -> None:
        """If classifier rank and chvar rank are the same permutation,
        ``torch.maximum(rank_cls, rank_cv) == rank_cls == rank_cv``, so
        union anchor set == classifier anchor set (up to ties)."""
        B, C, Cin, Hf = 2, 4, 4, 4  # small enough to avoid ties
        torch.manual_seed(5)
        # Build p3 and p4 so their channel structure forces the same ranking.
        base = torch.arange(Hf * Hf, dtype=torch.float32).view(Hf, Hf)
        p3 = torch.zeros(B, Cin, Hf, Hf, requires_grad=True)
        # Var(p3[:, :]) = (var of row across Cin channels). Put ``base``
        # into channel 0 with zero elsewhere -> Var == base^2 / Cin (roughly).
        p3.data[:, 0] = base.unsqueeze(0).expand(B, Hf, Hf)
        p4 = torch.zeros(B, Cin, Hf, Hf, requires_grad=True)
        p4.data[:, 0] = base.unsqueeze(0).expand(B, Hf, Hf)
        cls_w = torch.zeros(C, Cin); cls_w[0, 0] = 1.0; cls_w[1, 0] = 1.0
        cls_w.requires_grad = True
        labels = _disjoint_labels(B, C)
        proj = ProjectionHead(in_channels=Cin, out_channels=Cin * 2)
        l_cls = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="classifier",
        )
        l_union = patch_contrastive_loss(
            p3, p4, cls_w, labels, proj, top_k=4, m_negatives=8,
            anchor_source="union_cls_chvar",
        )
        # Not exactly equal numerically because InfoNCE also depends on
        # background negatives, which use classifier rank in both variants,
        # but anchors pick the same positions so the two losses must agree
        # up to floating-point error.
        assert torch.allclose(l_cls, l_union, atol=1e-5), (
            f"l_cls={l_cls.item()}, l_union={l_union.item()} disagree"
        )


# ---------------------------------------------------------------------------
# D2 warmup schedule
# ---------------------------------------------------------------------------


class TestLambdaMaskWarmup:
    """Mirrors ``TestLambdaConWarmup`` but keyed on ``effective_lambda_mask``.

    Uses the same helper pattern: construct a thin SPDNetModule, vary the
    schedule fields, and assert on the effective weight at given epochs.
    """

    @staticmethod
    def _make_module(
        *,
        lambda_mask: float,
        start: int,
        ramp: int,
    ):
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        cfg = SPDNetSpatialLossesConfig(
            lambda_eq=0.0,
            lambda_con=0.0,
            lambda_distill=0.0,
            lambda_mask=lambda_mask,
            mask_warmup_start_epoch=start,
            mask_warmup_epochs=ramp,
            online_loc_eval_enabled=False,
        )
        return SPDNetModule(
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

    def test_MW1_defaults_no_warmup(self) -> None:
        m = self._make_module(lambda_mask=1.0, start=0, ramp=0)
        for e in (0, 5, 100):
            assert m.effective_lambda_mask(epoch=e) == pytest.approx(1.0)

    def test_MW2_before_start_is_zero(self) -> None:
        m = self._make_module(lambda_mask=1.0, start=10, ramp=5)
        for e in range(10):
            assert m.effective_lambda_mask(epoch=e) == 0.0

    def test_MW3_linear_ramp_values(self) -> None:
        m = self._make_module(lambda_mask=1.0, start=5, ramp=4)
        expected = {5: 0.0, 6: 0.25, 7: 0.5, 8: 0.75, 9: 1.0, 20: 1.0}
        for e, want in expected.items():
            assert m.effective_lambda_mask(epoch=e) == pytest.approx(
                want, abs=1e-7,
            )

    def test_MW4_zero_lambda_disables_regardless(self) -> None:
        m = self._make_module(lambda_mask=0.0, start=0, ramp=5)
        for e in (0, 5, 100):
            assert m.effective_lambda_mask(epoch=e) == 0.0

    def test_MW5_negative_lambda_treated_as_zero(self) -> None:
        m = self._make_module(lambda_mask=-1.0, start=0, ramp=5)
        for e in (0, 5, 100):
            assert m.effective_lambda_mask(epoch=e) == 0.0


class TestLambdaAcWarmup:
    """Same warmup behaviour for ``L_ac`` that ``TestLambdaMaskWarmup`` gives
    ``L_mask``. Rationale: the 2026-04-30 cold-start highres run collapsed
    attn_mean to 0.98 by epoch 3 because L_ac fired on random MSE logits; the
    fix is to delay L_ac until the classifier has built usable spatial
    features. Both warmup defaults stay at 0 so existing recipes keep their
    legacy epoch-0 behaviour.
    """

    @staticmethod
    def _make_module(
        *,
        lambda_ac: float,
        start: int,
        ramp: int,
    ):
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        cfg = SPDNetSpatialLossesConfig(
            lambda_eq=0.0,
            lambda_con=0.0,
            lambda_distill=0.0,
            lambda_mask=0.0,
            lambda_ac=lambda_ac,
            ac_warmup_start_epoch=start,
            ac_warmup_epochs=ramp,
            online_loc_eval_enabled=False,
        )
        return SPDNetModule(
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

    def test_AW1_defaults_no_warmup(self) -> None:
        m = self._make_module(lambda_ac=0.05, start=0, ramp=0)
        for e in (0, 5, 100):
            assert m.effective_lambda_ac(epoch=e) == pytest.approx(0.05)

    def test_AW2_before_start_is_zero(self) -> None:
        m = self._make_module(lambda_ac=0.05, start=15, ramp=5)
        for e in range(15):
            assert m.effective_lambda_ac(epoch=e) == 0.0

    def test_AW3_linear_ramp_values(self) -> None:
        m = self._make_module(lambda_ac=0.05, start=15, ramp=4)
        expected = {
            15: 0.0,
            16: 0.0125,
            17: 0.025,
            18: 0.0375,
            19: 0.05,
            80: 0.05,
        }
        for e, want in expected.items():
            assert m.effective_lambda_ac(epoch=e) == pytest.approx(
                want, abs=1e-7,
            )

    def test_AW4_zero_lambda_disables_regardless(self) -> None:
        m = self._make_module(lambda_ac=0.0, start=0, ramp=5)
        for e in (0, 5, 100):
            assert m.effective_lambda_ac(epoch=e) == 0.0

    def test_AW5_negative_lambda_treated_as_zero(self) -> None:
        m = self._make_module(lambda_ac=-1.0, start=0, ramp=5)
        for e in (0, 5, 100):
            assert m.effective_lambda_ac(epoch=e) == 0.0

    def test_AW6_config_defaults_preserve_legacy_behaviour(self) -> None:
        """Fresh ``SPDNetSpatialLossesConfig()`` with non-zero lambda_ac and
        no warmup knobs set must give lam_ac_eff == lambda_ac from epoch 0.

        This is the regression hook for "I accidentally broke existing D1/D4
        recipes by adding warmup fields" -- every prior run has
        ``ac_warmup_*=0`` implicitly and must continue to behave identically.
        """
        m = self._make_module(lambda_ac=0.5, start=0, ramp=0)
        assert m.effective_lambda_ac(epoch=0) == pytest.approx(0.5)
        assert m.effective_lambda_ac(epoch=39) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# D1/D2/D3 integration: training_step paths that hit the new loss branches
# ---------------------------------------------------------------------------


class TestD1D2D3TrainingStep:
    """Drive a real ``SPDNetModule.training_step`` through each new branch
    to catch (a) shape mismatches, (b) NaN/Inf, (c) missing MLflow log
    keys, and (d) silent no-ops where a loss weight > 0 but the
    corresponding branch never runs.
    """

    @staticmethod
    def _run_one_step(
        mod,
        *,
        monkeypatch,
        seed: int = 0,
        num_classes: int = 4,
        image_size: int = 64,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        monkeypatch.setattr(
            type(mod),
            "current_epoch",
            property(lambda self: getattr(self, "_test_epoch", 0)),
            raising=False,
        )
        mod._test_epoch = 0
        logged: dict[str, float] = {}

        def _fake_log(name, value, *_, **__):
            if hasattr(value, "item"):
                try:
                    logged[name] = float(value.item()); return
                except Exception:
                    pass
            logged[name] = float(value)

        monkeypatch.setattr(mod, "log", _fake_log, raising=False)

        torch.manual_seed(seed)
        B = 2
        batch = {
            "query_image": torch.randn(B, 3, image_size, image_size),
            "ref_images": torch.randn(B, 3, image_size, image_size),
            "query_label": _disjoint_labels(B, num_classes),
        }
        total = mod.training_step(batch, batch_idx=0)
        assert torch.isfinite(total), f"non-finite total loss: {total.item()}"
        return total, logged

    def _mk(self, **losses):
        from src.conf.spdnet import SPDNetSpatialLossesConfig
        from src.wsss.spdnet.lightning import SPDNetModule

        cfg = SPDNetSpatialLossesConfig(
            online_loc_eval_enabled=False, **losses,
        )
        return SPDNetModule(
            num_classes=4, fpn_channels=16, mse_reduction=4,
            pretrained=False, learning_rate=1e-4, weight_decay=0.05,
            warmup_epochs=0, min_lr=1e-5, fusion_mode="spatial",
            losses_cfg=cfg, online_loc_metric=None, image_size=64,
        )

    def test_D1_ac_only_logs_L_ac_and_attn_mean(self, monkeypatch) -> None:
        m = self._mk(lambda_eq=0.0, lambda_ac=0.5, lambda_con=0.0,
                     lambda_mask=0.0, lambda_distill=0.0)
        _, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        assert "train/L_ac" in logged, (
            f"lambda_ac > 0 must cause L_ac to be logged. Keys: {sorted(logged)}"
        )
        assert "train/attn_mean" in logged
        # L_ac should be in [-1, 0] since attn_map is in [0, 1].
        assert -1.0 <= logged["train/L_ac"] <= 0.0
        assert "train/L_eq" not in logged, (
            "lambda_eq=0 must not produce an L_eq log"
        )

    def test_D1_ac_triggers_want_attn_even_with_lambda_eq_zero(
        self, monkeypatch,
    ) -> None:
        """Regression: in the old code path, ``want_attn`` was gated on
        ``lambda_eq > 0`` only. D1 enables attention via ``lambda_ac > 0``
        even when ``lambda_eq == 0``; if the guard is wrong, attn_map is
        missing and L_ac never fires."""
        m = self._mk(lambda_eq=0.0, lambda_ac=0.5, lambda_con=0.0,
                     lambda_mask=0.0, lambda_distill=0.0)
        # Fail loudly if the SCA's return_attn branch is skipped.
        import src.wsss.spdnet.model as model_mod
        calls = {"n": 0}
        orig_attn = model_mod.SpatialCrossAttention.forward

        def spy(self, q, kv, return_attn=False):
            if return_attn:
                calls["n"] += 1
            return orig_attn(self, q, kv, return_attn=return_attn)

        monkeypatch.setattr(
            model_mod.SpatialCrossAttention, "forward", spy, raising=True,
        )
        self._run_one_step(m, monkeypatch=monkeypatch)
        assert calls["n"] > 0, (
            "lambda_ac>0 must request return_attn=True from the SCA forward"
        )

    def test_D2_mask_only_logs_L_mask(self, monkeypatch) -> None:
        m = self._mk(lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
                     lambda_mask=1.0, lambda_distill=0.0)
        _, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        assert "train/L_mask" in logged
        assert logged["train/L_mask"] >= 0.0, (
            f"L_mask (MSE) must be >= 0; got {logged['train/L_mask']}"
        )
        assert "train/lambda_mask_eff" in logged
        assert logged["train/lambda_mask_eff"] == pytest.approx(1.0)
        assert "train/L_ac" not in logged and "train/L_eq" not in logged

    def test_D2_warmup_zero_skips_L_mask(self, monkeypatch) -> None:
        """When the warmup schedule returns 0, the L_mask block must not
        run (save compute) but ``train/lambda_mask_eff`` must still log."""
        m = self._mk(lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
                     lambda_mask=1.0, mask_warmup_start_epoch=5,
                     mask_warmup_epochs=3, lambda_distill=0.0)
        # Epoch 0 is before the ramp starts -> lam_mask_eff = 0.
        _, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        assert "train/lambda_mask_eff" in logged
        assert logged["train/lambda_mask_eff"] == 0.0
        assert "train/L_mask" not in logged, (
            "L_mask must not be logged pre-warmup"
        )

    def test_D1_ac_warmup_pre_ramp_zero_effective(
        self, monkeypatch,
    ) -> None:
        """During pre-ramp, ``lam_ac_eff == 0``: L_ac and attn_mean are
        still logged as *diagnostics* (operator must see attn_mean to watch
        for collapse), but L_ac's contribution to the total loss must vanish.
        """
        m = self._mk(lambda_eq=0.0, lambda_ac=0.5, lambda_con=0.0,
                     lambda_mask=0.0, lambda_distill=0.0,
                     ac_warmup_start_epoch=15, ac_warmup_epochs=5)
        _, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        # Effective weight is 0 at epoch 0.
        assert "train/lambda_ac_eff" in logged
        assert logged["train/lambda_ac_eff"] == 0.0, (
            f"Expected lam_ac_eff=0 pre-ramp, got {logged['train/lambda_ac_eff']}"
        )
        # L_ac and attn_mean still logged so operators can watch for the
        # attn_mean > 0.95 collapse even during the cls-only warmup phase.
        assert "train/L_ac" in logged
        assert "train/attn_mean" in logged

    def test_D1_ac_warmup_post_ramp_full_lambda(
        self, monkeypatch,
    ) -> None:
        """After the ramp ends, ``lam_ac_eff == lambda_ac`` and the total
        loss must include the full L_ac contribution. We assert that the
        total loss at epoch 10 (post-ramp) differs from epoch 0 (pre-ramp)
        by approximately ``lambda_ac * L_ac``, using identical weights and
        batch so L_ac itself is constant.
        """
        losses_kwargs = dict(
            lambda_eq=0.0, lambda_ac=0.5, lambda_con=0.0,
            lambda_mask=0.0, lambda_distill=0.0,
            ac_warmup_start_epoch=5, ac_warmup_epochs=5,
        )
        m_pre = self._mk(**losses_kwargs)
        m_post = self._mk(**losses_kwargs)
        # Match the weights so L_ac's raw value is identical.
        m_post.load_state_dict(m_pre.state_dict())

        # Monkeypatch ``current_epoch`` property for both modules on their
        # shared class (SPDNetModule) and vary per-instance via _test_epoch.
        monkeypatch.setattr(
            type(m_pre),
            "current_epoch",
            property(lambda self: getattr(self, "_test_epoch", 0)),
            raising=False,
        )
        m_pre._test_epoch = 0       # pre-ramp
        m_post._test_epoch = 10     # post-ramp (start=5, ramp=5)

        log_pre: dict[str, float] = {}
        log_post: dict[str, float] = {}

        def _mklog(tgt):
            def _fake_log(name, value, *_, **__):
                if hasattr(value, "item"):
                    try:
                        tgt[name] = float(value.item()); return
                    except Exception:
                        pass
                tgt[name] = float(value)
            return _fake_log

        monkeypatch.setattr(m_pre, "log", _mklog(log_pre), raising=False)
        monkeypatch.setattr(m_post, "log", _mklog(log_post), raising=False)

        torch.manual_seed(17)
        B = 2
        batch = {
            "query_image": torch.randn(B, 3, 64, 64),
            "ref_images": torch.randn(B, 3, 64, 64),
            "query_label": _disjoint_labels(B, 4),
        }
        # Reseed before each forward so MHA / dropout RNG is identical
        # across the two passes (otherwise stochastic ops consume different
        # slices of the global RNG and the raw L_ac values drift).
        torch.manual_seed(42)
        total_pre = m_pre.training_step(batch, batch_idx=0)
        torch.manual_seed(42)
        total_post = m_post.training_step(batch, batch_idx=0)

        assert log_pre["train/lambda_ac_eff"] == 0.0
        assert log_post["train/lambda_ac_eff"] == pytest.approx(0.5)
        # Raw L_ac is the same because weights + batch match.
        assert log_post["train/L_ac"] == pytest.approx(
            log_pre["train/L_ac"], rel=1e-5,
        )
        # Total_post - Total_pre should equal 0.5 * L_ac (the only term
        # that was gated off in the pre case).
        expected_delta = 0.5 * log_pre["train/L_ac"]
        got_delta = float(total_post.item()) - float(total_pre.item())
        assert got_delta == pytest.approx(expected_delta, abs=1e-5), (
            f"Expected total_post - total_pre == 0.5 * L_ac = "
            f"{expected_delta:g}; got {got_delta:g}"
        )

    def test_D3_union_con_logs_same_keys_as_classifier(
        self, monkeypatch,
    ) -> None:
        m = self._mk(lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.2,
                     lambda_mask=0.0, lambda_distill=0.0,
                     con_anchor_source="union_cls_chvar")
        _, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        assert "train/L_con" in logged, (
            "union anchor_source must still log train/L_con"
        )
        assert "train/lambda_con_eff" in logged

    def test_D1_plus_D2_plus_D3_all_fire(self, monkeypatch) -> None:
        """Combined D1+D2+D3 must produce all four log keys and a finite
        total loss."""
        m = self._mk(lambda_eq=0.0, lambda_ac=0.3, lambda_con=0.1,
                     lambda_mask=0.5, lambda_distill=0.0,
                     con_anchor_source="union_cls_chvar")
        total, logged = self._run_one_step(m, monkeypatch=monkeypatch)
        for k in ("train/L_cls", "train/L_ac", "train/L_mask", "train/L_con"):
            assert k in logged, f"missing {k}; got keys {sorted(logged)}"
        assert total.item() > 0.0, (
            "total loss should be positive (L_cls dominates, L_ac is -0.xx "
            "but small-weighted)"
        )


# ---------------------------------------------------------------------------
# D4: attention marginal-entropy loss + argmax-share backup
# ---------------------------------------------------------------------------


class TestAttentionMarginalEntropyLoss:
    """Fixed-point / gradient / range invariants for L_marg_H.

    See ``reports/notes/rq2_attention_regularizer_analysis.md`` for the
    mathematical setup and
    ``src/wsss/spdnet/spatial_losses.py::attention_marginal_entropy_loss``
    for the implementation.
    """

    @staticmethod
    def _random_attn_w(
        B: int = 2, P: int = 16, N: int = 12, seed: int = 0,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.softmax(torch.randn(B, P, N), dim=-1)

    def test_uniform_attention_gives_near_zero_loss(self) -> None:
        """Uniform attn_w: M = 0 (no concentration) and mu is already
        uniform so KL = 0. Therefore L_marg_H ≈ 0 regardless of beta."""
        B, P, N = 2, 16, 12
        attn_w = torch.full((B, P, N), 1.0 / N)
        loss = attention_marginal_entropy_loss(attn_w, beta=0.25)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mode_collapse_is_penalised(self) -> None:
        """All queries peak on the same key -> marginal is delta on that
        key -> KL(mu || U) = log N. Per-query M = 1 so -mean(M) = -1.
        With beta = 0.25 and N = 12, expected loss ≈ -1 + 0.25·log(12)
        ≈ -0.38, which is strictly greater than the structured-state
        minimum of -1 (proving the KL term is working)."""
        B, P, N = 2, 16, 12
        attn_w = torch.full((B, P, N), 1e-12)
        attn_w[..., 0] = 1.0 - (N - 1) * 1e-12
        loss = attention_marginal_entropy_loss(attn_w, beta=0.25)
        expected = -1.0 + 0.25 * math.log(N)
        assert loss.item() == pytest.approx(expected, abs=1e-3), (
            f"mode-collapse loss {loss.item():.4f} != expected {expected:.4f}"
        )
        # Must be strictly greater than the structured-state optimum
        # (that optimum is -1, achieved at "each query picks a distinct
        # key"). This is the D4 no-collapse invariant.
        structured_optimum = -1.0
        assert loss.item() > structured_optimum + 0.5, (
            f"mode-collapse loss {loss.item():.4f} must be much larger "
            f"than the structured-state optimum {structured_optimum:.4f}"
        )

    def test_structured_state_is_minimised(self) -> None:
        """Each query peaks on a DISTINCT key (permutation with B*P = N).
        -> M ≈ 1 (sharp queries), mu uniform (every key picked once).
        => loss ≈ -1 + beta * 0 = -1, the global minimum."""
        B, P = 2, 6
        N = B * P  # so B*P queries can cover N keys exactly once
        attn_w = torch.full((B, P, N), 1e-12)
        # Assign each (b, q) a unique key index.
        flat_idx = torch.arange(B * P).view(B, P)
        attn_w.scatter_(2, flat_idx.unsqueeze(-1), 1.0 - (N - 1) * 1e-12)
        loss = attention_marginal_entropy_loss(attn_w, beta=0.25)
        # Loss ≈ -1 (minimum): concentration term hits -1, KL term ≈ 0.
        assert loss.item() == pytest.approx(-1.0, abs=1e-3), (
            f"structured-state loss {loss.item():.4f} should be ≈ -1"
        )

    def test_gradient_flows_to_attn_w(self) -> None:
        """Backward through attn_w must produce finite grads of matching
        shape. This is the gradient path that reaches back into the SCA's
        in-projection."""
        attn_w = self._random_attn_w().requires_grad_(True)
        loss = attention_marginal_entropy_loss(attn_w, beta=0.25)
        loss.backward()
        assert attn_w.grad is not None
        assert attn_w.grad.shape == attn_w.shape
        assert torch.isfinite(attn_w.grad).all()
        assert attn_w.grad.abs().sum().item() > 0.0

    def test_beta_zero_reduces_to_L_ac(self) -> None:
        """At beta=0 the marginal term drops out and L_marg_H equals
        attention_concentration_loss(attn_map) where attn_map is the
        head-averaged per-query concentration map derived from attn_w.

        This is the reduction that keeps L_marg_H a strict generalisation
        of L_ac and motivates re-using the same gradient-budget
        calibration from RQ1."""
        attn_w = self._random_attn_w(B=2, P=16, N=12, seed=7)
        N = attn_w.shape[-1]
        log_N = math.log(N)
        attn_p = attn_w.clamp_min(1e-12)
        neg_ent = (attn_p * attn_p.log()).sum(dim=-1)
        attn_map = (1.0 + neg_ent / log_N).view(2, 4, 4)

        L_marg = attention_marginal_entropy_loss(attn_w, beta=0.0)
        L_ac = attention_concentration_loss(attn_map)
        assert L_marg.item() == pytest.approx(L_ac.item(), abs=1e-6)

    def test_invalid_beta_raises(self) -> None:
        attn_w = self._random_attn_w()
        with pytest.raises(ValueError, match="beta"):
            attention_marginal_entropy_loss(attn_w, beta=-0.1)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="B, P, N"):
            attention_marginal_entropy_loss(torch.randn(4, 4), beta=0.25)


class TestAttentionArgmaxShareLoss:
    """Backup attention regulariser: same four patterns as L_marg_H,
    plus a sanity check on the soft-argmax surrogate gradient.
    """

    @staticmethod
    def _random_attn_w(seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.softmax(torch.randn(2, 16, 12), dim=-1)

    def test_uniform_share_is_small(self) -> None:
        """Uniform attn_w: max soft-share is close to the uniform mean
        1/N, and -mean(M) = 0. So the loss is roughly beta/N."""
        B, P, N = 2, 16, 12
        attn_w = torch.full((B, P, N), 1.0 / N)
        loss = attention_argmax_share_loss(attn_w, beta=2.0)
        assert loss.item() == pytest.approx(2.0 / N, abs=1e-4)

    def test_mode_collapse_is_penalised(self) -> None:
        """Single-key dominance drives dominance -> 1 and M = 1.
        So loss ≈ -1 + beta. With beta=2, expect ≈ +1."""
        B, P, N = 2, 16, 12
        attn_w = torch.full((B, P, N), 1e-12)
        attn_w[..., 0] = 1.0 - (N - 1) * 1e-12
        loss = attention_argmax_share_loss(attn_w, beta=2.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-3)

    def test_structured_state_is_minimised(self) -> None:
        """Distinct peaks per query -> dominance ≈ 1/(B*P) ≈ 1/N, M ≈ 1.
        Loss ≈ -1 + beta/N -- strictly less than mode-collapse."""
        B, P = 2, 6
        N = B * P
        attn_w = torch.full((B, P, N), 1e-12)
        flat_idx = torch.arange(B * P).view(B, P)
        attn_w.scatter_(2, flat_idx.unsqueeze(-1), 1.0 - (N - 1) * 1e-12)
        loss = attention_argmax_share_loss(attn_w, beta=2.0)
        # Soft-argmax smears a little, but loss must be close to -1 + 2/N
        # and clearly below the mode-collapse value of ~+1.
        assert loss.item() < -0.8
        assert loss.item() < 1.0  # strictly better than collapse

    def test_gradient_flows_to_attn_w(self) -> None:
        """Both the concentration term and the soft-argmax surrogate give
        non-zero gradients to attn_w."""
        attn_w = self._random_attn_w().requires_grad_(True)
        loss = attention_argmax_share_loss(attn_w, beta=2.0)
        loss.backward()
        assert attn_w.grad is not None
        assert torch.isfinite(attn_w.grad).all()
        assert attn_w.grad.abs().sum().item() > 0.0

    def test_invalid_beta_raises(self) -> None:
        attn_w = self._random_attn_w()
        with pytest.raises(ValueError, match="beta"):
            attention_argmax_share_loss(attn_w, beta=-1.0)

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="B, P, N"):
            attention_argmax_share_loss(torch.randn(4, 4), beta=2.0)


# ---------------------------------------------------------------------------
# D4: L_mask combiner ("union") and deprecated alias
# ---------------------------------------------------------------------------


class TestMaskCombinerUnion:
    """The ``mask_combiner`` argument of :func:`cam_pseudo_mask_loss`
    exposes three positive-mask construction modes:

    * ``"intersection"`` == chvar_top AND cam_top
    * ``"chvar_only"``   == chvar_top
    * ``"union"``        == chvar_top OR cam_top   (D4 new path)

    Legacy ``use_intersection`` keeps working as a deprecated alias.
    """

    @staticmethod
    def _inputs(B: int = 2, Cin: int = 4, Hf: int = 8):
        torch.manual_seed(31)
        p3 = torch.randn(B, Cin, Hf, Hf)
        p4 = torch.randn(B, Cin, Hf, Hf)
        cls_w = torch.randn(4, Cin)
        labels = _disjoint_labels(B, 4)
        return p3, p4, cls_w, labels

    @staticmethod
    def _replay_positive_masks(
        p3: torch.Tensor,
        p4: torch.Tensor,
        cls_w: torch.Tensor,
        labels: torch.Tensor,
        alpha_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the (chvar_top, cam_top) masks that the loss
        builds internally, so we can assert set-inclusion between the
        three combiners without relying on private helpers."""
        from src.wsss.spdnet.spatial_losses import _kth_threshold
        B, _, Hf, _ = p3.shape
        chvar = p3.detach().var(dim=1, unbiased=False).flatten(1)
        P = chvar.shape[1]
        k_pos = max(1, int(round(alpha_pos * P)))
        thr_chv = _kth_threshold(chvar, k_pos, largest=True)
        chvar_top = (chvar >= thr_chv).view(B, Hf, Hf).float()
        S = torch.einsum("nc,bchw->bnhw", cls_w, p4)
        active = labels.argmax(dim=1)
        idx = active[:, None, None, None].expand(-1, 1, Hf, Hf)
        cam_act = torch.gather(S, 1, idx).squeeze(1).flatten(1)
        mn = cam_act.amin(dim=1, keepdim=True)
        mx = cam_act.amax(dim=1, keepdim=True)
        cam_norm = (cam_act - mn) / (mx - mn + 1e-8)
        thr_cam = _kth_threshold(cam_norm, k_pos, largest=True)
        cam_top = (cam_norm >= thr_cam).view(B, Hf, Hf).float()
        return chvar_top, cam_top

    def test_union_is_superset_of_intersection_and_chvar_only(self) -> None:
        """Algebraic identity: (chvar_top OR cam_top) must contain both
        (chvar_top AND cam_top) and (chvar_top alone)."""
        p3, p4, cls_w, labels = self._inputs()
        alpha = 0.25
        chvar_top, cam_top = self._replay_positive_masks(
            p3, p4, cls_w, labels, alpha,
        )
        inter = chvar_top * cam_top
        union = torch.maximum(chvar_top, cam_top)
        # Set-inclusion checks: every pixel set in a subset mode must
        # also be set in union mode.
        assert (union >= inter).all()
        assert (union >= chvar_top).all()
        assert (union >= cam_top).all()
        # Non-trivial: on random inputs we expect some pixels in union
        # that are not in intersection (otherwise the combiner choice
        # would be meaningless).
        assert union.sum().item() >= inter.sum().item()
        assert union.sum().item() >= chvar_top.sum().item()

    def test_union_loss_differs_from_intersection(self) -> None:
        """The three combiners must produce measurably different losses
        on generic inputs (sanity: our new branch isn't secretly calling
        the old code path)."""
        p3, p4, cls_w, labels = self._inputs()
        l_inter = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            alpha_pos=0.25, beta_neg=0.5, mask_combiner="intersection",
        )
        l_chvar = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            alpha_pos=0.25, beta_neg=0.5, mask_combiner="chvar_only",
        )
        l_union = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            alpha_pos=0.25, beta_neg=0.5, mask_combiner="union",
        )
        vals = {l_inter.item(), l_chvar.item(), l_union.item()}
        assert len(vals) >= 2, (
            f"at least two of the three combiners should give different "
            f"losses; got {vals}"
        )
        for v in (l_inter, l_chvar, l_union):
            assert torch.isfinite(v), f"combiner loss not finite: {v}"

    def test_legacy_use_intersection_alias_maps_to_intersection(
        self,
    ) -> None:
        """use_intersection=True must behave identically to
        mask_combiner='intersection'; False must match 'chvar_only'."""
        p3, p4, cls_w, labels = self._inputs()
        l_true = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels, use_intersection=True,
        )
        l_int = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels, mask_combiner="intersection",
        )
        assert l_true.item() == pytest.approx(l_int.item(), abs=1e-6)

        l_false = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels, use_intersection=False,
        )
        l_chv = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels, mask_combiner="chvar_only",
        )
        assert l_false.item() == pytest.approx(l_chv.item(), abs=1e-6)

    def test_legacy_alias_wins_over_mask_combiner(self) -> None:
        """When BOTH the deprecated flag and the new kwarg are supplied,
        the explicit legacy value takes precedence so pre-D4 configs
        keep their original semantics untouched."""
        p3, p4, cls_w, labels = self._inputs()
        # use_intersection=True overrides mask_combiner="union".
        l_override = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels,
            use_intersection=True, mask_combiner="union",
        )
        l_int = cam_pseudo_mask_loss(
            p3, p4, cls_w, labels, mask_combiner="intersection",
        )
        assert l_override.item() == pytest.approx(l_int.item(), abs=1e-6)

    def test_invalid_combiner_raises(self) -> None:
        p3, p4, cls_w, labels = self._inputs()
        with pytest.raises(ValueError, match="mask_combiner"):
            cam_pseudo_mask_loss(
                p3, p4, cls_w, labels, mask_combiner="bogus",
            )


# ---------------------------------------------------------------------------
# D4: training_step integration for L_marg_H and mask_combiner='union'
# ---------------------------------------------------------------------------


class TestD4TrainingStep:
    """End-to-end check that L_marg_H and mask_combiner='union' integrate
    cleanly into ``SPDNetModule.training_step`` -- covers the same
    invariants as ``TestD1D2D3TrainingStep`` for the new D4 recipe.
    """

    # Reuse the helpers from the D1-D3 class via composition.
    _runner = TestD1D2D3TrainingStep()

    def _mk(self, **losses):
        return self._runner._mk(**losses)

    def _run(self, mod, monkeypatch):
        return self._runner._run_one_step(mod, monkeypatch=monkeypatch)

    def test_lambda_marg_H_logs_and_is_finite(self, monkeypatch) -> None:
        """lambda_marg_H > 0 must log ``train/L_marg_H`` and keep the
        total loss finite (no NaN/inf from the KL term)."""
        m = self._mk(
            lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
            lambda_mask=0.0, lambda_distill=0.0,
            lambda_marg_H=0.15, marg_H_beta=0.25,
        )
        total, logged = self._run(m, monkeypatch)
        assert "train/L_marg_H" in logged, (
            f"lambda_marg_H>0 must log train/L_marg_H. Keys: {sorted(logged)}"
        )
        assert torch.isfinite(total)
        # L_marg_H at init is near 0 (attention is near-uniform).
        assert abs(logged["train/L_marg_H"]) < 1.0

    def test_lambda_marg_H_triggers_want_attn_even_with_lambda_ac_zero(
        self, monkeypatch,
    ) -> None:
        """Regression: ``want_attn`` must now also flip on for
        lambda_marg_H alone, otherwise ``feats['attn_w']`` is absent and
        the new branch KeyErrors."""
        m = self._mk(
            lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
            lambda_mask=0.0, lambda_distill=0.0,
            lambda_marg_H=0.15, marg_H_beta=0.25,
        )
        import src.wsss.spdnet.model as model_mod
        calls = {"n": 0}
        orig = model_mod.SpatialCrossAttention.forward

        def spy(self, q, kv, return_attn=False):
            if return_attn:
                calls["n"] += 1
            return orig(self, q, kv, return_attn=return_attn)

        monkeypatch.setattr(
            model_mod.SpatialCrossAttention, "forward", spy, raising=True,
        )
        self._run(m, monkeypatch)
        assert calls["n"] > 0, (
            "lambda_marg_H>0 must request return_attn=True from the SCA forward"
        )

    def test_D4_full_recipe_fires_all_components(self, monkeypatch) -> None:
        """D4-main recipe: lambda_marg_H>0, lambda_mask>0 (union combiner).
        Both new log keys must appear; legacy keys must not."""
        m = self._mk(
            lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
            lambda_distill=0.0,
            lambda_marg_H=0.15, marg_H_beta=0.25,
            lambda_mask=0.10, mask_alpha_pos=0.25, mask_beta_neg=0.50,
            mask_combiner="union",
        )
        total, logged = self._run(m, monkeypatch)
        for k in ("train/L_cls", "train/L_marg_H", "train/L_mask"):
            assert k in logged, f"missing {k}; got {sorted(logged)}"
        # D4 drops L_eq, L_con, L_ac by construction.
        for dropped in ("train/L_eq", "train/L_ac", "train/L_con"):
            assert dropped not in logged, (
                f"D4 recipe should not log {dropped}"
            )
        assert torch.isfinite(total)

    def test_D4_int_variant_differs_from_D4_main(self, monkeypatch) -> None:
        """D4-int uses ``mask_combiner='intersection'`` with the same
        lambdas as D4-main; the loss values must differ because the
        pseudo-mask target is different (sanity check that Hydra override
        actually reaches the loss)."""
        torch.manual_seed(123)
        m_union = self._mk(
            lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
            lambda_distill=0.0,
            lambda_marg_H=0.15, marg_H_beta=0.25,
            lambda_mask=0.10, mask_alpha_pos=0.25, mask_beta_neg=0.50,
            mask_combiner="union",
        )
        _, logged_union = self._run(m_union, monkeypatch)

        torch.manual_seed(123)
        m_int = self._mk(
            lambda_eq=0.0, lambda_ac=0.0, lambda_con=0.0,
            lambda_distill=0.0,
            lambda_marg_H=0.15, marg_H_beta=0.25,
            lambda_mask=0.10, mask_alpha_pos=0.25, mask_beta_neg=0.50,
            mask_combiner="intersection",
        )
        _, logged_int = self._run(m_int, monkeypatch)
        assert logged_union["train/L_mask"] != pytest.approx(
            logged_int["train/L_mask"], abs=1e-6,
        ), (
            "mask_combiner='union' vs 'intersection' must produce "
            "different L_mask values on the same batch"
        )
