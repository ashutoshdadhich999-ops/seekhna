"""
Energy analysis: converts measured operation counts into an energy
estimate, instead of a purely theoretical/asserted efficiency claim.

=== Method ===
For a conv/linear layer, the number of Multiply-Accumulate (MAC)
operations is a standard, exactly-countable quantity:

    MACs = output_elements * (kernel_elements * in_channels / groups)

An ANN layer performs one MAC per synaptic weight per forward pass. An
SNN layer instead performs an Accumulate (AC) -- a single addition, no
multiply -- *only when its input neuron spikes*. So the number of AC
operations a spiking layer actually performs is:

    ACs = MACs * spike_rate

where spike_rate is the *measured* fraction of active (spiking) inputs
(from src/evaluate.py::measure_sparsity), not an assumed constant. Energy
is then:

    Energy_ANN = MACs * E_MAC
    Energy_SNN = ACs   * E_AC   (for the spiking layers)
               + MACs  * E_MAC (for any non-spiking layers, e.g. the
                                 input/output conv, which are identical
                                 in both networks and never spike)

E_MAC and E_AC are taken from Horowitz, M. (2014), "Computing's Energy
Problem (and what we can do about it)", ISSCC -- a standard reference in
the SNN-efficiency literature for approximate 45nm CMOS per-operation
energies: E_MAC ~= 4.6 pJ (32-bit float multiply-add), E_AC ~= 0.9 pJ
(32-bit float add). These are widely-cited *approximate* figures, not a
measurement of this specific model on real hardware -- the resulting
numbers are an estimate of relative energy efficiency, not a guarantee of
real-device power draw. This limitation is stated explicitly wherever the
estimate is reported.
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import snntorch as snn


E_MAC_PJ = 4.6   # pJ per 32-bit MAC (Horowitz 2014, 45nm)
E_AC_PJ = 0.9    # pJ per 32-bit AC  (Horowitz 2014, 45nm)


@dataclass
class LayerOps:
    name: str
    macs: int
    is_spiking_input: bool  # True if this layer's input comes from a LIF spike


@dataclass
class EnergyReport:
    total_macs: int
    total_acs: int
    ann_energy_pj: float
    snn_energy_pj: float
    savings_pct: float
    per_layer: list = field(default_factory=list)

    def __str__(self):
        lines = [
            f"Total MACs (dense equivalent): {self.total_macs:,}",
            f"Total ACs (spike-gated):       {self.total_acs:,}",
            f"ANN energy estimate:  {self.ann_energy_pj / 1e6:.4f} uJ/sample "
            f"({self.ann_energy_pj:.1f} pJ)",
            f"SNN energy estimate:  {self.snn_energy_pj / 1e6:.4f} uJ/sample "
            f"({self.snn_energy_pj:.1f} pJ)",
            f"Estimated energy savings: {self.savings_pct:.1f}%",
        ]
        return "\n".join(lines)


def _count_macs_for_layer(module: nn.Module, output: torch.Tensor) -> int:
    """MACs for a single forward call, given the layer and its output."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        out_elems = output.shape[2:].numel() * output.shape[1]  # spatial * out_ch
        in_ch_per_group = module.in_channels // module.groups
        kernel_elems = 1
        for k in module.kernel_size:
            kernel_elems *= k
        macs_per_output = kernel_elems * in_ch_per_group
        return int(out_elems * macs_per_output)
    if isinstance(module, nn.Linear):
        return int(output.shape[-1] * module.in_features)
    return 0


def _is_spike_tensor(x: torch.Tensor) -> bool:
    """True if a tensor's values are (numerically) all 0 or 1, i.e. it is
    actually a spike train, not a dense/continuous activation. Used to
    decide, per layer and per forward call, whether that specific layer's
    input is spike-gated (AC-eligible) or not -- rather than assuming
    every layer in a "spiking" model is spike-fed, which is false for the
    first conv in every residual block (it reads the block's *dense*
    input, not a spike train; only the second conv in each block, which
    reads the LIF output, is genuinely spike-fed)."""
    if x.numel() == 0:
        return False
    return bool(torch.all((x == 0) | (x == 1)).item())


def estimate_energy(model: nn.Module, sample_input, sample_t, spike_rate: float = None,
                     is_spiking: bool = False) -> EnergyReport:
    """Run one forward pass, count MACs per conv/linear layer via hooks,
    and convert to an energy estimate.

    IMPORTANT (per-layer accounting): rather than applying one overall
    measured spike_rate uniformly to every layer of a spiking model, this
    inspects each layer's ACTUAL input tensor on this forward pass and
    only prices it as AC-eligible (spike-gated) if that specific input is
    a genuine 0/1 spike tensor. In every implemented spiking block here,
    only the second conv in each residual block reads spikes (the LIF
    output); the first conv in each block, and the input/output stem
    convs, read dense continuous activations and are always priced at
    full MAC cost, exactly as in the non-spiking model. This avoids
    silently giving a favorable AC discount to layers that never actually
    see a spike -- the uniform-rate version of this function did exactly
    that, which (if anything) *understated* the spiking model's true
    energy cost relative to what per-layer accounting gives.

    `spike_rate` is still accepted for backward compatibility / reporting
    but is no longer required for the energy math itself -- ACs are now
    computed directly from the measured per-layer active (nonzero)
    fraction of each spike-tensor input.
    """
    per_layer = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            macs = _count_macs_for_layer(module, out)
            layer_input = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else None
            spike_fed = is_spiking and layer_input is not None and _is_spike_tensor(layer_input)
            active_frac = 1.0
            if spike_fed and layer_input.numel() > 0:
                active_frac = float((layer_input != 0).float().mean().item())
            per_layer.append(LayerOps(name=name, macs=macs, is_spiking_input=spike_fed))
            per_layer[-1].__dict__["active_frac"] = active_frac
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(sample_input, sample_t)

    for h in hooks:
        h.remove()

    total_macs = sum(l.macs for l in per_layer)

    if is_spiking:
        total_acs = 0
        total_dense_macs = 0
        for l in per_layer:
            frac = l.__dict__.get("active_frac", 1.0)
            if l.is_spiking_input:
                total_acs += int(l.macs * frac)
            else:
                total_dense_macs += l.macs

        snn_energy = total_acs * E_AC_PJ + total_dense_macs * E_MAC_PJ
        ann_energy = total_macs * E_MAC_PJ  # dense-equivalent, for internal ratio only
        savings = 100.0 * (1 - snn_energy / max(ann_energy, 1e-8))
        return EnergyReport(total_macs, total_acs, ann_energy, snn_energy, savings, per_layer)
    else:
        ann_energy = total_macs * E_MAC_PJ
        return EnergyReport(total_macs, 0, ann_energy, ann_energy, 0.0, per_layer)


def compare_energy(spiking_model, nonspiking_model, sample_input, sample_t,
                    spike_rate: float) -> tuple:
    """Convenience wrapper: returns (spiking_report, nonspiking_report).

    Note: `snn_report.ann_energy_pj` is the SNN model's own dense-equivalent
    energy (same MAC count, as if it had no spiking sparsity) -- useful as
    an internal ratio, NOT the same as `nonspiking_model`'s actual energy.
    Use `real_world_savings()` below for the actual SNN-vs-ANN comparison.
    """
    snn_report = estimate_energy(spiking_model, sample_input, sample_t,
                                  spike_rate=spike_rate, is_spiking=True)
    ann_report = estimate_energy(nonspiking_model, sample_input, sample_t,
                                  is_spiking=False)
    return snn_report, ann_report


def real_world_savings(snn_report: EnergyReport, ann_report: EnergyReport) -> float:
    """Percentage energy saved by the actual spiking model vs. the actual
    matched non-spiking model (both real forward passes, not a
    same-architecture internal ratio)."""
    if ann_report.ann_energy_pj <= 0:
        return 0.0
    return 100.0 * (1 - snn_report.snn_energy_pj / ann_report.ann_energy_pj)
