# References

GaussianDog is an independent integration and implementation built from the following research and software.

## BITE and D-SMAL

- Nadine Rüegg, Shashank Tripathi, Konrad Schindler, Michael J. Black, and Silvia Zuffi. [BITE: Beyond Priors for Improved Three-D Dog Pose Estimation](https://openaccess.thecvf.com/content/CVPR2023/html/Ruegg_BITE_Beyond_Priors_for_Improved_Three-D_Dog_Pose_Estimation_CVPR_2023_paper.html). CVPR 2023.
- [Official BITE source](https://github.com/runa91/bite_release)
- [Official BITE project page](https://bite.is.tue.mpg.de/)

BITE introduces the dog-specific D-SMAL model and uses ground-contact information to improve pose reconstruction for standing, seated, and lying dogs.

## SMAL-pets

- Piotr Borycki, Joanna Waczyńska, Yizhe Zhu, Yongqiang Gao, and Przemysław Spurek. [SMAL-pets: SMAL Based Avatars of Pets from Single Image](https://arxiv.org/abs/2603.17131). 2026.
- [Official project page](https://piotr310100.github.io/SMAL-pets/)

The official project page currently labels its code as coming soon. GaussianDog independently implements the two-stage Gaussian optimization and surface-binding ideas described in the paper.

## 3D Gaussian Splatting

- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). SIGGRAPH 2023.
- [gsplat](https://github.com/nerfstudio-project/gsplat)

## TRELLIS

- Microsoft Research. [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506). CVPR 2025 Spotlight.
- [Official TRELLIS source](https://github.com/microsoft/TRELLIS)
- [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large)

## DGE

- Minghao Chen, Iro Laina, and Andrea Vedaldi. [DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing](https://arxiv.org/abs/2404.18929). ECCV 2024.
- [Official DGE source](https://github.com/silent-chen/DGE)

## Animation data used by the showcase application

- Lei Han et al. [Lifelike Agility and Play in Quadrupedal Robots using Reinforcement Learning and Generative Pre-trained Models](https://springernature.figshare.com/articles/dataset/Lifelike_Agility_and_Play_in_Quadrupedal_Robots_using_Reinforcement_Learning_and_Generative_Pre-trained_Models/24968946). The Labrador BVH data used for Walk and Run in Doggin' Around is licensed under CC BY 4.0.
