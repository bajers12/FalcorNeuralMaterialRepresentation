
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
        pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw one wo sample per batch element via the two-lobe GGX mixture.
        Returns:
            wo  : [B, 3]  sampled outgoing direction (world / shading space)
            pdf : [B]     probability density of wo under the mixture
        """

        pdf = self.eval_pdf()
        return wo, pdf

    def eval_pdf(
        self,
        z: torch.Tensor,
        wi: torch.Tensor,
        wo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the blended two-lobe GGX PDF p(wo | wi, z).
        p(wo) = sum_i  weight_i * p_i(wo)
        m is halfvector
        """

        wd, mu_dx, mu_dy, ws, ax, ay, rho, mus_x, mu_sy = self.forward(z, wi)

        # Half-vector m = normalize(wi + wo) in local frame
        m_local = self._safe_normalize(wi + wo)
        # Ensure m is on the upper hemisphere
        m_local = torch.where(
            m_local[..., 2:3] < 0.0, -m_local, m_local
        )

                             # p(wo)



