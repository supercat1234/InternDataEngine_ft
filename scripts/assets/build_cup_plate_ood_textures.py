#!/usr/bin/env python3
"""Reproducible procedural evaluation textures; no source assets are edited."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT/'local_configs/cup_plate_pi05_ood/textures'
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    y, x = np.mgrid[:512, :512]/512.
    grain = .5 + .25*np.sin(2*np.pi*(45*y + .65*np.sin(2*np.pi*x))) + .08*rng.normal(size=x.shape)
    walnut = np.stack([60+65*grain, 32+40*grain, 20+25*grain], axis=-1)
    Image.fromarray(np.uint8(np.clip(walnut,0,255))).save(folder/'walnut_grain.png')
    weave = .5+.16*np.sin(2*np.pi*64*x)+.16*np.sin(2*np.pi*64*y)
    woven = np.stack([105+35*weave, 112+35*weave, 120+35*weave], axis=-1)
    Image.fromarray(np.uint8(woven)).save(folder/'gray_woven.png')
    terrazzo = Image.new('RGB',(512,512),(220,218,205))
    draw = ImageDraw.Draw(terrazzo)
    for _ in range(1100):
        px,py=rng.integers(0,512,size=2); radius=int(rng.integers(1,5))
        colors=[(100,108,105),(155,120,100),(190,180,160),(242,240,228)]
        draw.ellipse((px-radius,py-radius,px+radius,py+radius), fill=colors[int(rng.integers(4))])
    terrazzo.save(folder/'terrazzo.png')
    for name, rgb in [('blue',(25,65,170)),('green',(35,145,80)),('terracotta',(175,70,35))]:
        Image.new('RGB',(64,64),rgb).save(folder/f'floor_{name}.png')


if __name__ == '__main__':
    main()
