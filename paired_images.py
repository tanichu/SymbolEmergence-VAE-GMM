"""Fixed paired RGB images; evaluation labels are never part of VAE inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class PairedImages(Dataset):
    def __init__(self,manifest,agent,split='train'):
        path=Path(manifest).resolve();data=json.loads(path.read_text())
        if agent not in ('a','b'):raise ValueError('Expected private agent a/b')
        self.rows=[r for r in data['pairs'] if r['split']==split]
        if not self.rows:raise ValueError('Empty requested split')
        ids={s:{r['scan_id'] for r in data['pairs'] if r['split']==s} for s in ('train','validation','test')}
        split_unit=data.get('split_unit','scan_id')
        if split_unit=='scan_id':
            if any(ids[a]&ids[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
                raise ValueError('Scan identity leaked across splits')
        elif split_unit=='camera_azimuth_group':
            poses={}
            for row in data['pairs']:
                for view in ['a','b']:
                    pose=row['pose_'+view]
                    key=(row['scan_id'],float(pose['azimuth'])%360,float(pose['elevation']))
                    if key in poses and poses[key]!=row['split']:
                        raise ValueError('Camera pose leaked across splits')
                    poses[key]=row['split']
        else:raise ValueError('Unknown split unit')
        images=[]
        for row in self.rows:
            p=(path.parent/row[agent]).resolve()
            if not p.is_relative_to(path.parent):raise ValueError('Image outside dataset root')
            image=Image.open(p).convert('RGB').resize((64,64),Image.Resampling.LANCZOS)
            images.append(torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255)
        self.images=torch.stack(images)
        self.labels=torch.tensor([r['label'] for r in self.rows],dtype=torch.long)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.images[index],self.labels[index]

def prepare_paired(manifest,batch_size):
    a,b=PairedImages(manifest,'a'),PairedImages(manifest,'b')
    assert len(a)==len(b) and torch.equal(a.labels,b.labels)
    loaders=[DataLoader(d,batch_size=batch_size,shuffle=False) for d in (a,b)]
    return *loaders,*loaders,len(a),list(range(len(a)))
