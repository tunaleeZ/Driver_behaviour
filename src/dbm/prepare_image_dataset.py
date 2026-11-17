"""Create stratified train/val/test folders from class-name image folders."""
import argparse, random, shutil
from pathlib import Path
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def prepare(source, output, seed=42, ratios=(.7,.15,.15)):
    source,output=Path(source),Path(output); rng=random.Random(seed)
    classes=sorted(p for p in source.iterdir() if p.is_dir())
    if len(classes)<2: raise ValueError(f'Expected class subfolders under {source}')
    for cls in classes:
        files=[p for p in cls.rglob('*') if p.suffix.lower() in EXTS]; rng.shuffle(files)
        if len(files)<3: raise ValueError(f'Class {cls.name} needs at least 3 images')
        n_train=max(1,int(len(files)*ratios[0])); n_val=max(1,int(len(files)*ratios[1]))
        if n_train+n_val>=len(files): n_train=len(files)-2; n_val=1
        groups={'train':files[:n_train],'val':files[n_train:n_train+n_val],'test':files[n_train+n_val:]}
        for split,items in groups.items():
            dest=output/split/cls.name; dest.mkdir(parents=True,exist_ok=True)
            for i,p in enumerate(items): shutil.copy2(p,dest/f'{p.stem}_{i}{p.suffix.lower()}')
    return output
def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); prepare(a.source,a.output,a.seed)
if __name__=='__main__': main()
