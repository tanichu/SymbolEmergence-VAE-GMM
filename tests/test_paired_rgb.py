import json
import pytest
import torch
from PIL import Image
from paired_images import PairedImages,prepare_paired
from vae_rgb64 import RGBVAE

def make_manifest(tmp_path,leak=False):
    for name,color in [('a',(255,0,0)),('b',(0,255,0))]:
        Image.new('RGB',(128,128),color).save(tmp_path/f'{name}.png')
    rows=[dict(scan_id='one',split='train',a='a.png',b='b.png',label=0),
          dict(scan_id='one' if leak else 'two',split='test',a='a.png',b='b.png',label=0)]
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(pairs=rows)))
    return path

def test_pair_order_pixels_and_no_mnist_rotation(tmp_path):
    manifest=make_manifest(tmp_path)
    a,b,all_a,all_b,n,indices=prepare_paired(manifest,4)
    x,label=next(iter(a));y,_=next(iter(b))
    assert x.shape==(1,3,64,64) and n==1 and indices==[0]
    assert torch.all(x[:,0]==1) and torch.all(x[:,1:]==0)
    assert torch.all(y[:,1]==1) and torch.all(y[:,0]==0)
    assert all_a is a and all_b is b

def test_scan_split_leak_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='leaked'):
        PairedImages(make_manifest(tmp_path,True),'a')

def test_rgb_shape_and_full_prior_gradient():
    torch.manual_seed(0)
    model=RGBVAE()
    image=torch.rand(2,3,64,64)
    reconstruction,mu,logvar,z=model(image)
    assert reconstruction.shape==image.shape and z.shape==(2,12)
    covariance=torch.eye(12).expand(2,12,12).clone()
    covariance[:,0,1]=covariance[:,1,0]=.15
    loss=model.loss_function(reconstruction,image,mu,logvar,torch.zeros(2,12),covariance,1)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
