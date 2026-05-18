
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
        raw[..., [0, 3]] = torch.softmax(raw[..., [0, 3]], dim=-1)

        return raw


    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        out = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _split_and_activate_params(
        self, pred: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if pred.ndim != 2 or pred.shape[-1] != 9:
            raise ValueError(f"Expected pred with shape [B, 9], got {tuple(pred.shape)}")

        wd = pred[..., 0]
        mu_dx = pred[..., 1]
        mu_dy = pred[..., 2]
        ws = pred[..., 3]
        ax_raw = pred[..., 4]
        ay_raw = pred[..., 5]
        rho_raw = pred[..., 6]
        mus_x = pred[..., 7]
        mu_sy = pred[..., 8]

        # Keep weights normalized even if pred is externally provided.
        w = torch.softmax(torch.stack([wd, ws], dim=-1), dim=-1)
        wd = w[..., 0]
        ws = w[..., 1]

        ax = self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * torch.sigmoid(ax_raw)
        ay = self.ALPHA_MIN + (self.ALPHA_MAX - self.ALPHA_MIN) * torch.sigmoid(ay_raw)
        rho = torch.tanh(rho_raw).clamp(min=-0.999, max=0.999)

        return wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy

    def _build_basis_from_normal(
        self, n: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_axis = torch.zeros_like(n)
        z_axis[..., 2] = 1.0
        x_axis = torch.zeros_like(n)
        x_axis[..., 0] = 1.0
        up = torch.where((n[..., 2:3].abs() > 0.999), x_axis, z_axis)

        t = self._safe_normalize(torch.cross(up, n, dim=-1))
        b = self._safe_normalize(torch.cross(n, t, dim=-1))
        return t, b

    def _sample_cosine_hemisphere(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u1 = torch.rand(batch_size, device=device, dtype=dtype)
        u2 = torch.rand(batch_size, device=device, dtype=dtype)

        r = torch.sqrt(u1.clamp(min=0.0, max=1.0))
        phi = 2.0 * math.pi * u2

        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = torch.sqrt((1.0 - u1).clamp_min(0.0))
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def _build_M(
        ax: torch.Tensor,
        ay: torch.Tensor,
        rho: torch.Tensor,
        mus_x: torch.Tensor,
        mu_sy: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros_like(ax)
        one = torch.ones_like(ax)
        s = torch.sqrt((1.0 - rho * rho).clamp_min(1e-8))

        row0 = torch.stack([ax, zero, -mus_x], dim=-1)
        row1 = torch.stack([ay * rho, ay * s, -mu_sy], dim=-1)
        row2 = torch.stack([zero, zero, one], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)


    def loss(
        self,
        pred: torch.Tensor,
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

        wo_sampled, pdf_q = self.sample(z, wi, pred)
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
        self,
        z: torch.Tensor,
        wi: torch.Tensor,
        pred: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw one wo sample per batch element via the two-lobe GGX mixture.
        Returns:
            wo  : [B, 3]  sampled outgoing direction (world / shading space)
            pdf : [B]     probability density of wo under the mixture
        """

        if pred is None:
            pred = self(z, wi)

        wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy = self._split_and_activate_params(pred)

        bsz = wi.shape[0]
        device = wi.device
        dtype = wi.dtype

        # Mixture sampling (Eq. 5): choose diffuse vs specular component.
        choose_spec = torch.rand(bsz, device=device, dtype=dtype) < ws

        # Diffuse branch (Eq. 6): cosine hemisphere sample tilted by n_d.
        nd = self._safe_normalize(
            torch.stack([-mu_dx, -mu_dy, torch.ones_like(mu_dx)], dim=-1)
        )
        td, bd = self._build_basis_from_normal(nd)
        wo_local = self._sample_cosine_hemisphere(bsz, device, dtype)
        wo_d = (
            td * wo_local[..., 0:1]
            + bd * wo_local[..., 1:2]
            + nd * wo_local[..., 2:3]
        )
        wo_d = self._safe_normalize(wo_d)

        # Specular branch (Eq. 9): sample h via transformed isotropic alpha=1 GGX,
        # then reflect wi around h.
        w_std = self._sample_cosine_hemisphere(bsz, device, dtype)
        M = self._build_M(ax, ay, rho, mus_x, mu_sy)
        wh = torch.matmul(M, w_std.unsqueeze(-1)).squeeze(-1)
        wh = self._safe_normalize(wh)
        wh = torch.where(wh[..., 2:3] < 0.0, -wh, wh)

        wi_dot_wh = (wi * wh).sum(dim=-1, keepdim=True)
        wo_s = 2.0 * wi_dot_wh * wh - wi
        wo_s = self._safe_normalize(wo_s)

        wo = torch.where(choose_spec.unsqueeze(-1), wo_s, wo_d)

        pdf = self.eval_pdf(
            wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy,
            wi=wi,
            wo=wo,
        )
        pdf = torch.nan_to_num(pdf, nan=1e-8, posinf=1e6, neginf=1e-8).clamp_min(1e-8)
        return wo, pdf

    def eval_pdf(
        self,
        wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy,
        wi: torch.Tensor,
        wo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the blended two-lobe GGX PDF p(wo | wi, z).
        p(wo) = sum_i  weight_i * p_i(wo)
        m is halfvector
        """
        eps = 1e-8

        # Ensure constraints if caller passes external values.
        w = torch.softmax(torch.stack([wd, ws], dim=-1), dim=-1)
        wd = w[..., 0]
        ws = w[..., 1]
        ax = ax.clamp(min=self.ALPHA_MIN, max=self.ALPHA_MAX)
        ay = ay.clamp(min=self.ALPHA_MIN, max=self.ALPHA_MAX)
        rho = rho.clamp(min=-0.999, max=0.999)

        # Diffuse lobe: cosine-weighted around tilted normal n_d.
        nd = self._safe_normalize(
            torch.stack([-mu_dx, -mu_dy, torch.ones_like(mu_dx)], dim=-1)
        )
        pd = (wo * nd).sum(dim=-1).clamp_min(0.0) / math.pi

        # Specular lobe (Eq. 7): evaluate transformed unit-roughness GGX density.
        wh = self._safe_normalize(wi + wo)
        wh = torch.where(wh[..., 2:3] < 0.0, -wh, wh)

        M = self._build_M(ax, ay, rho, mus_x, mu_sy)
        invM = torch.linalg.inv(M)

        v = torch.matmul(invM, wh.unsqueeze(-1)).squeeze(-1)
        v_len = v.norm(dim=-1).clamp_min(eps)
        v_hat = v / v_len.unsqueeze(-1)

        # For alpha=1 GGX: projected normal distribution equals cos(theta)/pi.
        D_std = v_hat[..., 2].clamp_min(0.0) / math.pi

        det_inv = torch.linalg.det(invM).abs().clamp_min(eps)
        p_wh = D_std * det_inv / (v_len ** 3)

        jac = (4.0 * (wo * wh).sum(dim=-1).abs()).clamp_min(eps)
        ps = p_wh / jac

        valid = (
            (wi[..., 2] > 0.0)
            & (wo[..., 2] > 0.0)
            & ((wi * wh).sum(dim=-1) > 0.0)
            & ((wo * wh).sum(dim=-1) > 0.0)
        )
        ps = torch.where(valid, ps, torch.zeros_like(ps))

        p = wd * pd + ws * ps
        return torch.nan_to_num(p, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)



