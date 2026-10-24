"""Run with Blender --background --factory-startup --disable-autoexec --python."""
import argparse,json,math,sys
from pathlib import Path
import bpy
from mathutils import Vector,Matrix

parser=argparse.ArgumentParser()
parser.add_argument('--pack',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--probe',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
pack=Path(args.pack);out=Path(args.output)
out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Refuse dataset overwrite')
selection={'Apple':(['A34','A59','A73'],'A81','A99'),
           'Orange':(['A49','A61','A62'],'A64','A67'),
           'Mango':(['A29','A57','A114'],'A115','A130')}
manifest=dict(protocol='Same physical scan, different camera views; development pilot',
    resolution=128,source_lod='mid',pairs=[],classes=list(selection),render_engine='CYCLES',samples=16)
for label,(category,(train,val,test)) in enumerate(selection.items()):
    for ident in train+[val,test]:
        if args.probe and ident!='A34':continue
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        model_files=list(pack.glob(f'*_Mid/Models/{ident}_*_mid.fbx'))
        if len(model_files)!=1:raise RuntimeError(f'Ambiguous source {ident}: {model_files}')
        bpy.ops.import_scene.fbx(filepath=str(model_files[0]))
        objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
        if not objects:raise RuntimeError('No meshes imported')
        corners=[o.matrix_world@Vector(c) for o in objects for c in o.bound_box]
        low=Vector(tuple(min(c[k] for c in corners) for k in range(3)))
        high=Vector(tuple(max(c[k] for c in corners) for k in range(3)))
        center=(low+high)/2;scale=2/max(high-low)
        for obj in objects:obj.matrix_world=Matrix.Scale(scale,4)@Matrix.Translation(-center)@obj.matrix_world
        # Explicit albedo hookup avoids stale packaged absolute texture paths.
        textures=list(pack.glob(f'*_Mid/Textures/{ident}_*/*albedo*'))
        if not textures:textures=[p for p in pack.glob('*_Mid/Textures/**/*') if p.is_file() and p.name.startswith(ident+'_') and 'albedo' in p.name.lower()]
        if len(textures)!=1:raise RuntimeError(f'Missing/ambiguous albedo for {ident}: {textures}')
        material=bpy.data.materials.new(ident+'_verified_albedo');material.use_nodes=True
        bsdf=material.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.6
        tex=material.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(textures[0]),check_existing=True)
        material.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color'])
        for obj in objects:obj.data.materials.clear();obj.data.materials.append(material)
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
        scene.render.threads_mode='FIXED';scene.render.threads=4
        scene.render.resolution_x=scene.render.resolution_y=128;scene.render.resolution_percentage=100
        scene.view_settings.view_transform='Standard'
        scene.world=bpy.data.worlds.new('Neutral world');scene.world.use_nodes=True
        scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.65,.65,.65,1)
        scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.7
        for position,power,size in [((3,-4,5),350,4),((-4,-1,3),200,4)]:
            light=bpy.data.lights.new('area','AREA');light.energy=power;light.shape='DISK';light.size=size
            obj=bpy.data.objects.new('area',light);scene.collection.objects.link(obj);obj.location=position
            obj.rotation_euler=(-obj.location).to_track_quat('-Z','Y').to_euler()
        camera=bpy.data.cameras.new('camera');camera.type='ORTHO';camera.ortho_scale=3.1
        cam=bpy.data.objects.new('camera',camera);scene.collection.objects.link(cam);scene.camera=cam
        split='train' if ident in train else 'validation' if ident==val else 'test'
        for elevation in ([20] if args.probe else [20,40]):
            for azimuth in ([0] if args.probe else range(0,360,30)):
                row=dict(scan_id=ident,label=label,category=category,split=split,
                         source_fbx=str(model_files[0]),base_azimuth=azimuth,base_elevation=elevation)
                for agent,da,de in [('a',0,0),('b',60,15)]:
                    az,el=math.radians(azimuth+da),math.radians(elevation+de)
                    cam.location=(5*math.cos(el)*math.cos(az),5*math.cos(el)*math.sin(az),5*math.sin(el))
                    cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler()
                    filename=f'{ident}/az{azimuth:03d}-el{elevation:02d}-{agent}.png'
                    path=out/filename;path.parent.mkdir(exist_ok=True)
                    if path.exists():raise FileExistsError(path)
                    scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
                    row[agent]=filename
                manifest['pairs'].append(row)
                print('PAIR_DONE',ident,azimuth,elevation,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('MANIFEST_DONE',len(manifest['pairs']),flush=True)
