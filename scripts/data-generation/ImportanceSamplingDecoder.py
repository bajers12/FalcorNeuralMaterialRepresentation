
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from BrdfDecoder import Decoder

class ImportanceSamplingDecoder(nn.Module):
    """
    Importance sampling decoder matching the paper (Section 4.3 / Figure 4):
      - The network predicts parameters of a two-lobe anisotropic GGX (Trowbridge-Reitz)
        microfacet distribution

      - Sampling follows the standard GGX visible-normal (VNDF) path:
          1. Sample a microfacet normal m ~ GGX-NDF in the lobe's shading frame.
          2. Reflect wi around m to get wo.
          3. Evaluate the blended two-lobe PDF at wo.
      - The MLP only sees wi (not wo), consistent with Fig. 4 ("ωi" feeds the sampler).
    """

    # Roughness is clamped to [alpha_min, 1] to avoid singularities in the GGX NDF.
    ALPHA_MIN: float = 0.01
    ALPHA_MAX: float = 1.0

    def __init__(
        self,
        latent_ch: int,
        mlp_width: int = 32,
        mlp_depth: int = 2,
        use_bias_in_mlp: bool = True,
    ):
        super().__init__()

        self.latent_ch = latent_ch


        # MLP inputs: latent z  +  raw incident direction wi (3 components)
        mlp_in = latent_ch + 3
        layers: list[nn.Module] = []
        prev = mlp_in
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width, bias=use_bias_in_mlp))
            layers.append(nn.ReLU(inplace=True))
            prev = mlp_width
        layers.append(nn.Linear(prev, 9, bias=use_bias_in_mlp))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, wi: torch.Tensor):
        """
        Predict per-lobe GGX parameters from latent code z and incident direction wi.

        Returns:
            {wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy}
            wd and wp are diffue and specular weights
            mu_dx and mu_dy are surface slope parameters for cosine weighted normal tilt
            ax and ay are orthgogonal rougness values
            rho is correlation value for 2 above
            mu_sx and mu_sx surface slope parameters for NDF mean offset
        """

        # Feed raw wi directly to MLP (not frame-projected)
        # {wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy}
        raw = self.mlp(torch.cat([z, wi], dim=-1))  # [B, 9]

        #raw lobe weight -> softmax mixture
        torch._softmax(raw[..., [0,3]], dim=-1)

        return raw



    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        out = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _ggx_ndf(m_local: torch.Tensor,
                 alpha_x: torch.Tensor,
                 alpha_y: torch.Tensor) -> torch.Tensor:
        """
        Anisotropic GGX (Trowbridge-Reitz) NDF evaluated at a half-vector h.
        D(h) = 1 / (π α_x α_y  ((hx/αx)² + (hy/αy)² + hz²)²)
        The half-vector is already available in the lobe-local frame, so use
        its components directly instead of reconstructing hx/hy through
        sqrt(1 - cos(theta)^2). The direct form is equivalent and has a much
        better behaved gradient at normal incidence.
        """
        ax = alpha_x.clamp_min(1e-6)
        ay = alpha_y.clamp_min(1e-6)
        hx = m_local[..., 0]
        hy = m_local[..., 1]
        hz = m_local[..., 2].clamp_min(0.0)

        denom_sq = ((hx / ax) ** 2 + (hy / ay) ** 2 + hz ** 2) ** 2
        return 1.0 / (math.pi * ax * ay * denom_sq.clamp_min(1e-10))

    @staticmethod
    def _ggx_smith_g1(v_local: torch.Tensor,
                      alpha_x: torch.Tensor,
                      alpha_y: torch.Tensor) -> torch.Tensor:
        """
        Anisotropic GGX Smith G1 masking term.
        Uses the local direction components directly:
        G1(v) = 2 / (1 + sqrt(1 + ((alpha_x vx)^2 + (alpha_y vy)^2) / vz^2)).
        """
        ax = alpha_x.clamp_min(1e-6)
        ay = alpha_y.clamp_min(1e-6)
        vx = v_local[..., 0]
        vy = v_local[..., 1]
        vz = v_local[..., 2].clamp_min(1e-6)

        tan2_alpha2 = ((ax * vx) ** 2 + (ay * vy) ** 2) / (vz ** 2)
        return 2.0 / (1.0 + (1.0 + tan2_alpha2.clamp_min(0.0)).sqrt())

    # ------------------------------------------------------------------
    # Two-lobe GGX sampling (Heitz 2018 visible-normal sampling per lobe)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_ggx_vndf_single_lobe(
        wi_local: torch.Tensor,   # [B, 3] in lobe's local frame (z == N)
        alpha_x: torch.Tensor,    # [B]
        alpha_y: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        Sample a microfacet normal m from the GGX VNDF (Heitz 2018) in local
        space where the lobe normal is the z-axis.
        Returns m [B, 3] (unit, z >= 0 in the lobe frame).

        Algorithm from: Sampling the GGX Distribution of Visible Normals,
        Heitz 2018, JCGT.
        """
        device = wi_local.device
        dtype = wi_local.dtype
        B = wi_local.shape[0]

        # Step 1 — stretch wi by roughness
        wi_s = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([
                wi_local[..., 0] * alpha_x,
                wi_local[..., 1] * alpha_y,
                wi_local[..., 2].clamp_min(1e-6),
            ], dim=-1)
        )

        # Step 2 — build orthonormal basis (t1, t2) around wi_s
        sign = torch.where(wi_s[..., 2] >= 0.0,
                           torch.ones(B, device=device, dtype=dtype),
                           -torch.ones(B, device=device, dtype=dtype))
        a = -1.0 / (sign + wi_s[..., 2]).clamp_min(1e-6)
        b_coeff = wi_s[..., 0] * wi_s[..., 1] * a
        t1 = torch.stack([
            1.0 + sign * wi_s[..., 0] ** 2 * a,
            sign * b_coeff,
            -sign * wi_s[..., 0],
        ], dim=-1)
        t2 = torch.stack([
            b_coeff,
            sign + wi_s[..., 1] ** 2 * a,
            -wi_s[..., 1],
        ], dim=-1)

        # Step 3 — sample point on disk parameterization
        u1 = torch.rand(B, device=device, dtype=dtype)
        u2 = torch.rand(B, device=device, dtype=dtype)

        r = u1.sqrt()
        phi = 2.0 * math.pi * u2
        t = r * torch.cos(phi)
        s = r * torch.sin(phi)
        t = t + s * (1.0 - t.abs()) * (
            torch.where(wi_s[..., 2] >= 0.0,
                        torch.zeros_like(s),
                        (1.0 - t.abs()) / (1.0 - wi_s[..., 2].abs()).clamp_min(1e-6))
        )

        # Step 4 — reproject onto hemisphere
        mh = (
            t.unsqueeze(-1) * t1
            + s.unsqueeze(-1) * t2
            + (1.0 - t**2 - s**2).clamp_min(0.0).sqrt().unsqueeze(-1) * wi_s
        )

        # Step 5 — un-stretch
        m = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([
                mh[..., 0] * alpha_x,
                mh[..., 1] * alpha_y,
                mh[..., 2].clamp_min(0.0),
            ], dim=-1)
        )
        return m

    @staticmethod
    def _ggx_vndf_pdf(
        wi_local: torch.Tensor,   # [B, 3]
        m_local: torch.Tensor,    # [B, 3]  microfacet normal in lobe frame
        alpha_x: torch.Tensor,    # [B]
        alpha_y: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        PDF of the GGX VNDF: p(m) = G1(wi) D(m) |wi·m| / |wi·n|
        where n is the macro-surface normal (z-axis in local frame).
        Returns p(m) [B].  The reflection Jacobian |∂m/∂wo| = 1/(4|wo·m|)
        is applied by the caller to get p(wo).
        """
        cos_theta_i = wi_local[..., 2].clamp_min(1e-6)

        D = ImportanceSamplingDecoder._ggx_ndf(m_local, alpha_x, alpha_y)
        G1 = ImportanceSamplingDecoder._ggx_smith_g1(wi_local, alpha_x, alpha_y)
        wi_dot_m = (wi_local * m_local).sum(dim=-1).clamp_min(1e-6)
        return G1 * D * wi_dot_m / cos_theta_i.clamp_min(1e-6)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def loss(
        self,
        decoder: Decoder,
        z: torch.Tensor,
        wi: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        KL divergence loss for the importance sampler (Section 4.3).

        Minimises:  KL(q || p̃) = E_{wo ~ q(wo|wi,z)} [ log q(wo|wi,z) - log p̃(wo|wi,z) ]

        where:
            q(wo|wi,z) : the sampler's two-lobe GGX-mixture PDF
            p̃(wo|wi,z) : unnormalised target ∝ f(wi,wo)·cos(θo), evaluated by the
                        frozen BRDF decoder.  cos(θo) is the dataset convention
                        (targets already include it) so the decoder output is used
                        directly as the unnormalised weight.
        """
        # --- 1. Single forward_params call — reused for both sample and eval_pdf ---
        x = self.forward(z, wi)

        # --- 2. Sample wo on-policy WITH gradient (reparameterized), reusing cached frames ---
        wo_sampled, pdf_q = self.sample(
            z, wi
        )
        pdf_q = torch.nan_to_num(pdf_q, nan=eps, posinf=1e6, neginf=eps).clamp(min=eps, max=1e6)
        log_q = torch.log(pdf_q)  # [B], has grad through wo_sampled -> frames/alpha

        # --- 3. Evaluate frozen BRDF target at sampled wo ---
        with torch.no_grad():
            # wo_sampled has grad but is safe to use inside no_grad for value computation;
            # log_p_tilde is fully detached — only log_q carries the gradient.
            y_hat = decoder.forward(z, wi, wo_sampled)  # [B, 3]  (f(wi,wo))
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)

            brdf_luminance = (
                0.2126 * y_hat[..., 0]
                + 0.7152 * y_hat[..., 1]
                + 0.0722 * y_hat[..., 2]
            ).clamp_min(eps)  # [B]
            p_tilde = (brdf_luminance).clamp_min(eps)

            log_p_tilde = torch.log(p_tilde)  # fully detached

        # --- 4. Direct KL loss: E_{wo~q}[ log q - log p̃ ] ---
        loss_terms = log_q - log_p_tilde  # [B], grad through log_q only

        finite_mask = torch.isfinite(loss_terms)
        if not finite_mask.any():
            return (pdf_q * 0.0).sum()

        return loss_terms[finite_mask].mean()

    def sample(
        self, z: torch.Tensor, wi: torch.Tensor,
        _frames: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw one wo sample per batch element via the two-lobe GGX mixture.
        Returns:
            wo  : [B, 3]  sampled outgoing direction (world / shading space)
            pdf : [B]     probability density of wo under the mixture
        """
        if _frames is None:
            t, bv, n, alpha, lobe_weights = self.forward_params(z, wi)
        else:
            t, bv, n, alpha, lobe_weights = _frames
        B = z.shape[0]

        # --- Select which lobe to use for each sample (stratified by weight) ---
        lobe_idx = torch.multinomial(lobe_weights, num_samples=1).squeeze(-1)  # [B]

        # Gather the chosen lobe's frame and roughness
        idx = lobe_idx.view(B, 1, 1)
        t_sel  = t.gather(1,  idx.expand(B, 1, 3)).squeeze(1)   # [B, 3]
        bv_sel = bv.gather(1, idx.expand(B, 1, 3)).squeeze(1)
        n_sel  = n.gather(1,  idx.expand(B, 1, 3)).squeeze(1)
        alpha_sel = alpha.gather(1, lobe_idx.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)  # [B, 2]
        ax = alpha_sel[..., 0]
        ay = alpha_sel[..., 1]

        # Transform wi into the selected lobe's local frame
        wi_local = torch.stack([
            (wi * t_sel).sum(-1),
            (wi * bv_sel).sum(-1),
            (wi * n_sel).sum(-1),
        ], dim=-1)  # [B, 3]

        # Flip wi below the hemisphere to the upper side (back-facing case)
        wi_local = torch.where(
            wi_local[..., 2:3] < 0.0,
            wi_local * torch.tensor([1.0, 1.0, -1.0], device=wi.device, dtype=wi.dtype),
            wi_local,
        )

        # Sample microfacet normal m in local frame, then reflect
        m_local = self._sample_ggx_vndf_single_lobe(wi_local, ax, ay)      # [B, 3]
        wo_local = 2.0 * (wi_local * m_local).sum(-1, keepdim=True) * m_local - wi_local

        # Transform wo back to world / shading space
        wo = (
            wo_local[..., 0:1] * t_sel
            + wo_local[..., 1:2] * bv_sel
            + wo_local[..., 2:3] * n_sel
        )
        wo = self._safe_normalize(wo)

        # Evaluate the blended PDF at the sampled wo
        pdf = self.eval_pdf(z, wi, wo, _frames=(t, bv, n, alpha, lobe_weights))
        return wo, pdf

    def eval_pdf(
        self,
        z: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
        _frames: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Evaluate the blended two-lobe GGX PDF p(wo | wi, z).
        p(wo) = sum_i  weight_i * p_i(wo)
        where p_i(wo) = p_vndf_i(m_i) / (4 |wo · m_i|)  and  m_i = normalize(wi + wo).
        """

        alpha, lobe_weights = self.forward(z, wi)




        # Half-vector m = normalize(wi + wo) in local frame
        m_local = self._safe_normalize(wi + wo)
        # Ensure m is on the upper hemisphere
        m_local = torch.where(
            m_local[..., 2:3] < 0.0, -m_local, m_local
        )

        wo_dot_m = (wo * m_local).sum(-1).abs().clamp_min(1e-6)

        p_m = self._ggx_vndf_pdf(wi, m_local, ax_i, ay_i)  # p(m)
        p_wo = p_m / (4.0 * wo_dot_m)                            # p(wo)

        p_wo = torch.nan_to_num(p_wo, nan=0.0, posinf=0.0, neginf=0.0)
        w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)

        pdf_total = pdf_total + w_i * p_wo.clamp_min(0.0)

        return torch.nan_to_num(pdf_total, nan=1e-10, posinf=1e10, neginf=1e-10).clamp_min(1e-10)
