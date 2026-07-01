
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
        return self.mlp(torch.cat([z, wi], dim=-1))  # [B, 9]


    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        out = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _tanh_approx(x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(1.0 + x * x)

    @staticmethod
    def _sinh_approx(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(1.0 + x * x)

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

        alpha_x_raw = pred[..., 0]
        alpha_y_raw = pred[..., 1]
        rho_raw = pred[..., 2]
        slope_spec_x_raw = pred[..., 3]
        slope_spec_y_raw = pred[..., 4]
        slope_diff_x_raw = pred[..., 5]
        slope_diff_y_raw = pred[..., 6]
        w_spec_raw = pred[..., 7]
        w_diff_raw = pred[..., 8]

        alpha_x = 1e-4 + 0.5 * (1.0 + self._tanh_approx(alpha_x_raw))
        alpha_y = 1e-4 + 0.5 * (1.0 + self._tanh_approx(alpha_y_raw))
        rho = self._tanh_approx(rho_raw).clamp(min=-0.999, max=0.999)
        slope_spec_x = self._sinh_approx(slope_spec_x_raw)
        slope_spec_y = self._sinh_approx(slope_spec_y_raw)
        slope_diff_x = self._sinh_approx(slope_diff_x_raw)
        slope_diff_y = self._sinh_approx(slope_diff_y_raw)

        weights = torch.softmax(torch.stack([w_spec_raw, w_diff_raw], dim=-1), dim=-1)
        w_spec = weights[..., 0]
        w_diff = weights[..., 1]

        return alpha_x, alpha_y, rho, slope_spec_x, slope_spec_y, slope_diff_x, slope_diff_y, w_spec, w_diff

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
    def _sample_diffuse(
        wo_local: torch.Tensor, slope_diff_x: torch.Tensor, slope_diff_y: torch.Tensor
    ) -> torch.Tensor:
        n = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([-slope_diff_x, -slope_diff_y, torch.ones_like(slope_diff_x)], dim=-1)
        )
        up = torch.zeros_like(n)
        up[..., 2] = 1.0
        fallback = torch.zeros_like(n)
        fallback[..., 0] = 1.0
        helper = torch.where((n[..., 2:3].abs() > 0.999), fallback, up)
        t = ImportanceSamplingDecoder._safe_normalize(torch.cross(helper, n, dim=-1))
        b = ImportanceSamplingDecoder._safe_normalize(torch.cross(n, t, dim=-1))
        return (
            t * wo_local[..., 0:1]
            + b * wo_local[..., 1:2]
            + n * wo_local[..., 2:3]
        )

    @staticmethod
    def _sample_specular_half_vector(
        u: torch.Tensor,
        alpha_x: torch.Tensor,
        alpha_y: torch.Tensor,
        rho: torch.Tensor,
        slope_spec_x: torch.Tensor,
        slope_spec_y: torch.Tensor,
    ) -> torch.Tensor:
        u1 = u[..., 0].clamp(1e-6, 1.0 - 1e-6)
        u2 = u[..., 1].clamp(1e-6, 1.0 - 1e-6)
        s = torch.sqrt(u1) / torch.sqrt((1.0 - u1).clamp_min(1e-6))
        phi = 2.0 * math.pi * u2
        sx_std = s * torch.cos(phi)
        sy_std = s * torch.sin(phi)

        sqrt_one_minus_rho = torch.sqrt((1.0 - rho * rho).clamp_min(1e-8))
        sx = alpha_x * sx_std
        sy = alpha_y * (rho * sx_std + sqrt_one_minus_rho * sy_std)
        sx = sx + slope_spec_x
        sy = sy + slope_spec_y

        return ImportanceSamplingDecoder._safe_normalize(
            torch.stack([-sx, -sy, torch.ones_like(sx)], dim=-1)
        )

    @staticmethod
    def _pdf_diffuse(wo: torch.Tensor, slope_diff_x: torch.Tensor, slope_diff_y: torch.Tensor) -> torch.Tensor:
        n = ImportanceSamplingDecoder._safe_normalize(
            torch.stack([-slope_diff_x, -slope_diff_y, torch.ones_like(slope_diff_x)], dim=-1)
        )
        return (wo * n).sum(dim=-1).clamp_min(0.0) / math.pi

    @staticmethod
    def _pdf_specular(
        wi: torch.Tensor,
        wo: torch.Tensor,
        alpha_x: torch.Tensor,
        alpha_y: torch.Tensor,
        rho: torch.Tensor,
        slope_spec_x: torch.Tensor,
        slope_spec_y: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        wh = ImportanceSamplingDecoder._safe_normalize(wi + wo)
        wh = torch.where(wh[..., 2:3] < 0.0, -wh, wh)

        cos_theta = wh[..., 2].clamp_min(0.0)
        valid = (cos_theta > 1e-4) & ((wi * wh).sum(dim=-1) > 0.0) & ((wo * wh).sum(dim=-1) > 0.0)

        sx = -wh[..., 0] / cos_theta.clamp_min(eps)
        sy = -wh[..., 1] / cos_theta.clamp_min(eps)
        sx = sx - slope_spec_x
        sy = sy - slope_spec_y

        sx_std = sx / alpha_x.clamp_min(eps)
        sqrt_one_minus_rho = torch.sqrt((1.0 - rho * rho).clamp_min(eps))
        normalization = 1.0 / (alpha_x.clamp_min(eps) * alpha_y.clamp_min(eps) * sqrt_one_minus_rho)
        sy_std = (alpha_x * sy - rho * alpha_y * sx) * normalization

        r2 = sx_std * sx_std + sy_std * sy_std
        p22_std = 1.0 / (math.pi * (1.0 + r2) * (1.0 + r2))
        p22 = p22_std * normalization
        pdf_h = p22 / torch.clamp(cos_theta * cos_theta * cos_theta, min=eps)
        jac = 4.0 * (wi * wh).sum(dim=-1).abs().clamp_min(eps)
        ps = pdf_h / jac
        return torch.where(valid, ps, torch.zeros_like(ps))


    def loss(
        self,
        pred: torch.Tensor,
        decoder: Decoder,
        z: torch.Tensor,
        wi: torch.Tensor,
        eps: float = 1e-6,
        target_grad: bool = False,
        loss_order: str = "q_minus_p",
        discard_below_surface: bool = False,
        horizon_eps: float = 0.0,
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
        if target_grad:
            # Keep decoder weights fixed while allowing gradients through wo_sampled.
            decoder_params = list(decoder.parameters())
            previous_requires_grad = [p.requires_grad for p in decoder_params]
            for p in decoder_params:
                p.requires_grad_(False)
            try:
                y_hat = decoder.forward(z, wi, wo_sampled)  # [B, 3]  (f(wi,wo))
            finally:
                for p, requires_grad in zip(decoder_params, previous_requires_grad):
                    p.requires_grad_(requires_grad)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
        else:
            with torch.no_grad():
                y_hat = decoder.forward(z, wi, wo_sampled)  # [B, 3]  (f(wi,wo))
                y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)

        brdf_luminance = (
            0.2126 * y_hat[..., 0]
            + 0.7152 * y_hat[..., 1]
            + 0.0722 * y_hat[..., 2]
        ).clamp_min(eps)  # [B]
        p_tilde = (brdf_luminance).clamp_min(eps)
        log_p_tilde = torch.log(p_tilde)

        # --- 4. Direct KL loss: E_{wo~q}[ log q - log p̃ ] ---
        if loss_order == "q_minus_p":
            loss_terms = log_q - log_p_tilde  # [B]
        elif loss_order == "p_minus_q":
            loss_terms = log_p_tilde - log_q  # [B]
        else:
            raise ValueError(f"Unknown sampler loss order: {loss_order}")

        finite_mask = torch.isfinite(loss_terms)
        if discard_below_surface:
            finite_mask = finite_mask & (wo_sampled[..., 2] > horizon_eps)
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

        alpha_x, alpha_y, rho, slope_spec_x, slope_spec_y, slope_diff_x, slope_diff_y, w_spec, w_diff = self._split_and_activate_params(pred)

        bsz = wi.shape[0]
        device = wi.device
        dtype = wi.dtype

        # Mixture sampling (Eq. 5): choose diffuse vs specular component.
        u = torch.rand(bsz, 2, device=device, dtype=dtype)
        choose_spec = u[..., 0] < w_spec
        u_select = torch.where(
            choose_spec,
            u[..., 0] / w_spec.clamp_min(1e-6),
            (u[..., 0] - w_spec) / w_diff.clamp_min(1e-6),
        ).clamp(1e-6, 1.0 - 1e-6)
        u_local = torch.stack([u_select, u[..., 1]], dim=-1)

        # Diffuse branch (Eq. 6): cosine hemisphere sample tilted by n_d.
        wo_diff_local = self._sample_cosine_hemisphere(bsz, device, dtype)
        wo_d = self._sample_diffuse(wo_diff_local, slope_diff_x, slope_diff_y)
        wo_d = self._safe_normalize(wo_d)

        # Specular branch (Listing 4): slope-space GGX sample and reflection.
        wh = self._sample_specular_half_vector(
            u_local, alpha_x, alpha_y, rho, slope_spec_x, slope_spec_y
        )
        wi_dot_wh = (wi * wh).sum(dim=-1, keepdim=True)
        wo_s = 2.0 * wi_dot_wh * wh - wi
        wo_s = self._safe_normalize(wo_s)

        wo = torch.where(choose_spec.unsqueeze(-1), wo_s, wo_d)

        pdf = self.eval_pdf(
            alpha_x, alpha_y, rho, slope_spec_x, slope_spec_y, slope_diff_x, slope_diff_y, w_spec, w_diff,
            wi=wi,
            wo=wo,
        )
        pdf = torch.nan_to_num(pdf, nan=1e-8, posinf=1e6, neginf=1e-8).clamp_min(1e-8)
        return wo, pdf

    def eval_pdf(
        self,
        alpha_x, alpha_y, rho, slope_spec_x, slope_spec_y, slope_diff_x, slope_diff_y, w_spec, w_diff,
        wi: torch.Tensor,
        wo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the blended two-lobe GGX PDF p(wo | wi, z).
        p(wo) = sum_i  weight_i * p_i(wo)
        m is halfvector
        """
        eps = 1e-8

        alpha_x = alpha_x.clamp(min=1e-4, max=1.0)
        alpha_y = alpha_y.clamp(min=1e-4, max=1.0)
        rho = rho.clamp(min=-0.999, max=0.999)
        w_spec = w_spec.clamp(min=0.0, max=1.0)
        w_diff = w_diff.clamp(min=0.0, max=1.0)

        # Diffuse PDF: cosine-weighted around tilted normal n_d.
        nd = self._safe_normalize(
            torch.stack([-slope_diff_x, -slope_diff_y, torch.ones_like(slope_diff_x)], dim=-1)
        )
        pd = (wo * nd).sum(dim=-1).clamp_min(0.0) / math.pi

        # Specular PDF: slope-space GGX model from the article.
        wh = self._safe_normalize(wi + wo)
        wh = torch.where(wh[..., 2:3] >= 0.0, wh, -wh)
        cos_theta = wh[..., 2]
        valid = (cos_theta > 1e-4) & ((wi * wh).sum(dim=-1) > 0.0) & ((wo * wh).sum(dim=-1) > 0.0)

        sx = -wh[..., 0] / cos_theta.clamp_min(eps)
        sy = -wh[..., 1] / cos_theta.clamp_min(eps)
        sx = sx - slope_spec_x
        sy = sy - slope_spec_y

        sx_std = sx / alpha_x
        sqrt_one_minus_rho = torch.sqrt((1.0 - rho * rho).clamp_min(eps))
        normalization = 1.0 / (alpha_x * alpha_y * sqrt_one_minus_rho).clamp_min(eps)
        sy_std = (alpha_x * sy - rho * alpha_y * sx) * normalization

        r2 = sx_std * sx_std + sy_std * sy_std
        p22_std = 1.0 / (math.pi * (1.0 + r2) * (1.0 + r2))
        p22 = p22_std * normalization
        pdf_h = p22 / torch.clamp(cos_theta * cos_theta * cos_theta, min=eps)
        jac = (4.0 * (wi * wh).sum(dim=-1).abs()).clamp_min(eps)
        ps = pdf_h / jac

        p = w_spec * ps + w_diff * pd
        p = torch.where(valid, p, w_diff * pd)
        return torch.nan_to_num(p, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)



