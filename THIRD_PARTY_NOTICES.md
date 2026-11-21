# Third-party notices

The root MIT license covers original ThinkWAM code. The following components
retain their upstream terms; the root license does not replace those terms.

| Component | Scope | Upstream license |
|---|---|---|
| [V-JEPA 2 / 2.1](https://github.com/facebookresearch/vjepa2) | Vendored files in `thinkwam_p0/models/vjepa/` and pretrained visual encoder | [MIT](licenses/V-JEPA-MIT.txt), Meta Platforms, Inc. and affiliates |
| [JEPA-WAM](https://github.com/SpriteWithoutIce/JEPA_WAM) | Adapted encoder/projector/robot-policy contracts; upstream attribution retained | [MIT](licenses/JEPA-WAM-MIT.txt), copyright holders listed in that notice |
| [LeJEPA](https://github.com/galilai-group/lejepa) | `thinkwam_p0/models/sigreg.py`, adapted sliced Epps–Pulley quadrature and ThinkWAM DDP/masking/token-pooling extensions | [CC BY-NC 4.0](licenses/LeJEPA-CC-BY-NC-4.0.txt) |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | External pretrained language model, including its tensors in released policy checkpoints | [Apache 2.0](licenses/Qwen-Apache-2.0.txt) |

**The SIGReg component is subject to a noncommercial license.** Do not interpret
the root MIT license as removing that restriction. Its upstream source is
[LeJEPA MINIMAL.md](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md),
by Randall Balestriero and Yann LeCun; ThinkWAM adds distributed statistics,
mask handling and pooling. No endorsement by the upstream authors is implied.

Published policy checkpoints contain ThinkWAM parameters plus V-JEPA and Qwen
parameters. Embedded upstream weights retain their MIT and Apache 2.0
notices respectively; the code license does not replace checkpoint licensing.
SIGReg has no persistent checkpoint parameters. Datasets and simulators are distributed
separately and remain subject to their own licenses.
