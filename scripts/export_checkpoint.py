"""Export inference states without optimizer, paths, or experiment tracker metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models import Encoder, Decoder, ARModel


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def export(source, config, output):
    source_state=torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    c=OmegaConf.load(config)
    states={key:{name.removeprefix('_orig_mod.'): tensor for name,tensor in source_state[key].items()}
            for key in ('enc','dec','ar_model','ar_model_ema')}
    for cls,key in ((Encoder,'enc'),(Decoder,'dec'),(ARModel,'ar_model'),(ARModel,'ar_model_ema')):
        # Assignment checks all keys/shapes without allocating a second model's storage.
        with torch.device('meta'): model=cls(c)
        model.load_state_dict(states[key],strict=True,assign=True)
        del model
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    torch.save(dict(format_version=1,**states),out)
    saved=torch.load(out,map_location='cpu',weights_only=True,mmap=True)
    for key,state in states.items():
        for name,tensor in state.items():
            if not torch.equal(tensor,saved[key][name]): raise AssertionError((key,name))
    return dict(filename=out.name,sha256=sha256(out),bytes=out.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); result=export(a.source,a.config,a.output)
    Path(a.output+'.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
