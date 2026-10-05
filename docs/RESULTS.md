# Huawei Tricolor result

The showcase run started from Huawei's seated Tricolor dog photograph and produced a standing, animatable Gaussian reconstruction.

| Input photograph | Standing reference | Final Gaussian dog |
|:---:|:---:|:---:|
| <img src="images/huawei-tricolor-input.png" width="230" alt="Huawei Tricolor dog input"> | <img src="images/standing-reference.png" width="230" alt="Standing reference"> | <img src="images/gaussian-dog-output.png" width="330" alt="Final Gaussian dog"> |

## Reconstruction

| Measurement | Result |
|---|---:|
| Bound optimization | 15,000 steps |
| Free optimization | 25,000 steps |
| Optional DGE pass | 1,000 steps over 20 views |
| Pre-DGE Gaussians | 37,926 |
| Accepted Gaussians | 37,525 |
| Held-out PSNR before DGE | 43.9274 |
| Held-out SSIM before DGE | 0.996132 |
| Synthetic views | 96 at 512 by 512 |
| Held-out views | 8 |

The full DGE result completed, but it softened tan markings and eye detail. The accepted appearance kept the original face and body Gaussians and retained only the improved tail geometry with a corrected dark coat color. This is why the accepted count is smaller than the pre-DGE model.

## Deformation and motion

| Check | Result |
|---|---:|
| Binding rest identity error | 5.96e-8 |
| Binding weight-sum error | 1.19e-7 |
| Full vertex recovery error | 1.79e-7 |
| Maximum paw and sole contact error | 0.00563 viewer units |
| Maximum floor penetration | 0.001965 viewer units |
| Contact and floor acceptance gate | 0.013 viewer units |
| Animation sampling | 30 FPS |
| Actions | 12 |

The animations include a front-first jump sequence and a digging loop with alternating front-paw scrapes and planted hind paws.

## Browser rendering

The complete 37,525-Gaussian appearance measured 60 FPS in five samples at a 1440 by 960 WebGL drawing buffer in Chrome on the test Mac. Runtime performance depends on GPU, pixel ratio, camera framing, and Gaussian count.

## Compute

The full temporary A6000 session lasted 4 hours and 55 minutes. Its conservative combined compute and storage estimate was US$2.96. All required results were downloaded before the pod was terminated.
