# -*- coding: utf-8 -*-
"""Try to swap the skeletal mesh of the 'target' blueprint to the Girl_01 meshes
(the original L-map look) via the reflection bridge; tile the results."""
import os, sys, time, json
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import uz_env as U

TGT = '/Game/HumanCharacter/ThirdPersonBP/Blueprints/target.target'
MESHES = ['/Game/HumanCharacter/Girl_01/meshes/girl_01_a.girl_01_a', '/Game/HumanCharacter/Girl_01/meshes/girl_01_e.girl_01_e',
          '/Game/HumanCharacter/Girl_01/meshes/girl_01_f.girl_01_f', '/Game/HumanCharacter/Girl_01/meshes/girl_01_i.girl_01_i',
          '/Game/HumanCharacter/girl2/claire.claire', '/Game/HumanCharacter/girl3/lola_b_styperek.lola_b_styperek']
OUT = os.path.join(HERE, '..', 'results', 'logs', 'uz_room')
env = U.UZEnv('FlexibleRoom', sleep_time=30); c = env.c


def q(cmd, n=600):
    return str(c.request(cmd, timeout=30))[:n].replace('\n', ' ')


try:
    print('help ->', q('vreflect help', 1500))
    base = (-450.0, -1500.0); z = 178.0
    c.request(f'vset /objects/spawn_from_path {TGT} t0')
    c.request([f'vset /object/t0/location {base[0]} {base[1]} {z}', 'vset /object/t0/rotation 0 -90 0', 'vbp t0 set_phy 0'], -1)
    c.request(f'vset /camera/0/location {base[0]} {base[1] - 250} {z + 20}'); c.request('vset /camera/0/rotation -5 90 0')
    time.sleep(0.8)
    for cmd in ['vreflect t0 components', 'vreflect t0 list_components', 'vreflect t0 list_properties', 'vreflect t0 list_functions']:
        print(cmd, '->', q(cmd, 900))
    tiles = []
    for i, m in enumerate(MESHES):
        tried = []
        for cmd in [f'vreflect t0 Mesh set SkeletalMesh {m}', f'vreflect t0.Mesh set SkeletalMesh {m}', f'vreflect t0 call SetSkeletalMesh {m}',
                    f'vreflect t0.Mesh call SetSkeletalMesh {m}', f'vreflect t0.CharacterMesh0 call SetSkeletalMesh {m}',
                    f'vreflect t0.CharacterMesh0 set SkeletalMesh {m}', f'vreflect t0.CharacterMesh0 call SetSkeletalMeshAsset {m}']:
            r = q(cmd, 160); tried.append((cmd.split(' ', 2)[1] + ' ' + cmd.split(' ')[2], r))
            if 'error' not in r.lower() and 'fail' not in r.lower():
                break
        time.sleep(0.5)
        im = env.api.get_image(0, 'lit')[60:420, 220:420].copy()
        cv2.putText(im, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2); tiles.append(im)
        print(m.split('/')[-1], '->', tried[-1])
    cv2.imwrite(os.path.join(OUT, 'char_mesh.png'), np.hstack(tiles)); print('saved char_mesh.png')
finally:
    env.close()
