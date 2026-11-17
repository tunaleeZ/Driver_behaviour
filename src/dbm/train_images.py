"""Train frame-wise CNN from split/class/image folder datasets."""
from __future__ import annotations
import argparse, csv, json, logging
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
from dbm.models.cnn_image import CNNImageClassifier
from dbm.metrics import classification_metrics
from dbm.seed import set_global_seed

LOG=logging.getLogger(__name__)
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
class FolderDataset(Dataset):
    def __init__(self, root, classes, train, size):
        self.samples=[]; self.train=train; self.size=size
        for i,name in enumerate(classes):
            folder=Path(root)/name
            if not folder.is_dir(): raise FileNotFoundError(f'Missing class directory: {folder}')
            self.samples += [(p,i) for p in folder.rglob('*') if p.suffix.lower() in EXTS]
        if not self.samples: raise ValueError(f'No images in {root}')
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,y=self.samples[i]
        image=cv2.imread(str(p))
        if image is None: raise ValueError(f'Cannot decode image: {p}')
        image=cv2.cvtColor(cv2.resize(image,(self.size,self.size)),cv2.COLOR_BGR2RGB)
        if self.train:
            if np.random.rand()<.5: image=cv2.flip(image,1)
            image=np.clip((image.astype(np.float32)-127.5)*np.random.uniform(.85,1.15)+127.5,0,255)
            image=np.clip(image*np.random.uniform(.85,1.15),0,255).astype(np.uint8)
        x=torch.from_numpy(image.copy()).permute(2,0,1).float()/255
        mean=torch.tensor((.485,.456,.406))[:,None,None]; std=torch.tensor((.229,.224,.225))[:,None,None]
        return (x-mean)/std,y

def run(data, output, epochs=20, batch=32, size=224, lr=3e-4, seed=42, workers=2):
    set_global_seed(seed); data=Path(data); output=Path(output)
    classes=sorted(p.name for p in (data/'train').iterdir() if p.is_dir())
    if len(classes)<2: raise ValueError('Need at least two class folders under train/')
    sets={s:FolderDataset(data/s,classes,s=='train',size) for s in ('train','val','test')}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders={s:DataLoader(ds,batch_size=batch,shuffle=s=='train',num_workers=workers,pin_memory=device.type=='cuda') for s,ds in sets.items()}
    counts=np.bincount([y for _,y in sets['train'].samples],minlength=len(classes)); weights=torch.tensor(len(sets['train'])/(len(classes)*np.maximum(counts,1)),dtype=torch.float32,device=device)
    model=CNNImageClassifier(n_classes=len(classes)).to(device); loss_fn=nn.CrossEntropyLoss(weight=weights); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    output.mkdir(parents=True,exist_ok=True); best=-1; stale=0; history=[]
    for epoch in range(1,epochs+1):
        row={'epoch':epoch}
        for split in ('train','val'):
            model.train(split=='train'); ys=[]; ps=[]; total=0; loss_sum=0
            for x,y in loaders[split]:
                x=x.to(device); y=y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split=='train'):
                    logits=model(x); loss=loss_fn(logits,y)
                    if split=='train': loss.backward(); opt.step()
                total+=len(y); loss_sum+=float(loss.detach())*len(y); ys.extend(y.cpu().tolist()); ps.extend(logits.detach().argmax(1).cpu().tolist())
            metric=classification_metrics(ys,ps,len(classes)); row[f'{split}_loss']=loss_sum/max(total,1); row[f'{split}_macro_f1']=metric['macro_f1']
        history.append(row); LOG.info('epoch %d val_macro_f1 %.4f',epoch,row['val_macro_f1'])
        if row['val_macro_f1']>best:
            best=row['val_macro_f1']; stale=0
            torch.save({'model':model.state_dict(),'model_type':'cnn_image','config':{'model_type':'cnn_image','classes':classes,'input_size':size,'cnn_channels':[32,64,128,256],'target_fps':5},'classes':classes,'epoch':epoch},output/'best.pt')
        else: stale+=1
        if stale>=7: break
    with (output/'epochs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    ckpt=torch.load(output/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ckpt['model']); model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loaders['test']:
            ps.extend(model(x.to(device)).argmax(1).cpu().tolist()); ys.extend(y.tolist())
    result={'classes':classes,'test':classification_metrics(ys,ps,len(classes))}; (output/'results.json').write_text(json.dumps(result,indent=2)); return output

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True,help='root containing train/val/test/class_name/images'); p.add_argument('--output',default='outputs/cnn_image'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--size',type=int,default=224); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--workers',type=int,default=2); a=p.parse_args(); logging.basicConfig(level=logging.INFO); run(a.data,a.output,a.epochs,a.batch_size,a.size,a.lr,workers=a.workers)
if __name__=='__main__': main()
