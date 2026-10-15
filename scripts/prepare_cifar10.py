"""Convert the official CIFAR-10 training split into the wARTok PNG layout."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-dir',default='./data/downloads')
    p.add_argument('--output',default='./data/cifar10')
    p.add_argument('--download',action='store_true')
    p.add_argument('--reference-images',help='Optional NPZ of the same 50K training images for ADM statistics')
    a=p.parse_args(); out=Path(a.output)
    if out.exists() and any(out.iterdir()): p.error('output must be empty to prevent mixing splits')
    ds=CIFAR10(a.download_dir,train=True,download=a.download)
    assert len(ds)==50000
    labels=[]
    for i,(image,label) in enumerate(ds):
        name=f'{i//1000:05d}/img{i:08d}.png'
        path=out/name; path.parent.mkdir(parents=True,exist_ok=True)
        image.save(path); labels.append([name,label])
    (out/'dataset.json').write_text(json.dumps({'labels':labels}))
    if a.reference_images:
        with open(a.reference_images,'xb') as f: np.savez_compressed(f,ds.data)
    print(f'Wrote {len(ds)} training images to {out}')


if __name__=='__main__': main()
