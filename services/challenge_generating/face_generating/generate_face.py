from typing import Optional, Tuple
import click
from services.challenge_generating.face_generating import dnnlib
import numpy as np
import PIL.Image
import torch
import random
import legacy


class FaceGenerator:
    def __init__(self):
        self.G = self.init_model("https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl")
    def __call__(self):
        image = self.generate_images(self.G)
        image = image.resize((512,512))
        return image
    def init_model(self, network_pkl):
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        return G
    def generate_images(
        self,
        G,
        truncation_psi: float = 1,
        noise_mode: str = "const",
        translate: Tuple[float,float] = (0, 0),
        rotate: float = 0,
        class_idx: Optional[int]=0
    ):
        seeds = [random.randint(1, 1e9)]
        device = torch.device('cuda')
        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                raise click.ClickException('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            # Construct an inverse rotation/translation matrix and pass to the generator.  The
            # generator expects this matrix as an inverse to avoid potentially failing numerical
            # operations in the network.
            if hasattr(G.synthesis, 'input'):
                m = self.make_transform(translate, rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))

            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
            return img

    def make_transform(self, translate: Tuple[float,float], angle: float):
        m = np.eye(3)
        s = np.sin(angle/360.0*np.pi*2)
        c = np.cos(angle/360.0*np.pi*2)
        m[0][0] = c
        m[0][1] = s
        m[0][2] = translate[0]
        m[1][0] = -s
        m[1][1] = c
        m[1][2] = translate[1]
        return m


if __name__ == "__main__":
    import time
