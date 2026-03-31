import sys, os, numpy as np, starfile
sys.path.insert(0, r'c:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify')
os.chdir(r'c:\Users\LongChen\Documents\ResearchRelated\Dev\Agent\NucC2Align260218_simplify\experiments\PolysomeManual_1_Noise')
from linker_prediction.models import Particle
from linker_prediction.star_utils import pick_cols, colvec, euler_zyz_to_Zaxis
from scipy.spatial import cKDTree

pixel_size_nm = 1.96 / 10.0
data = starfile.read('Avg_Linkers.star', always_dict=True)
df = next(iter(data.values()))
df = df.rename(columns={c: c.strip() for c in df.columns})

C0 = pick_cols(df, ['rlnLC_CoordinateX0','rlnLC_CoordinateY0','rlnLC_CoordinateZ0'])
C1 = pick_cols(df, ['rlnLC_CoordinateX1','rlnLC_CoordinateY1','rlnLC_CoordinateZ1'])
A1 = pick_cols(df, ['rlnLC_AngleRot1','rlnLC_AngleTilt1','rlnLC_AnglePsi1'])
C2 = pick_cols(df, ['rlnLC_CoordinateX2','rlnLC_CoordinateY2','rlnLC_CoordinateZ2'])
A2 = pick_cols(df, ['rlnLC_AngleRot2','rlnLC_AngleTilt2','rlnLC_AnglePsi2'])

cen = colvec(df, C0)*pixel_size_nm
a1v = colvec(df, C1)*pixel_size_nm
a2v = colvec(df, C2)*pixel_size_nm
t1 = euler_zyz_to_Zaxis(colvec(df, A1))
t2 = euler_zyz_to_Zaxis(colvec(df, A2))

nucs = [Particle(center=cen[i], a1=a1v[i], a2=a2v[i], t1=t1[i], t2=t2[i]) for i in range(len(df))]

# How far apart are a1 and a2 within the same particle?
spreads = [np.linalg.norm(nucs[i].a1 - nucs[i].a2) for i in range(len(nucs))]
print(f'Arm spread (a1-a2 within same particle):')
print(f'  min={min(spreads):.2f}  max={max(spreads):.2f}  mean={np.mean(spreads):.2f} nm')

# KDTree candidate generation (original: all arms)
pts = np.empty((2*len(nucs), 3))
for i in range(len(nucs)):
    pts[2*i] = nucs[i].a1
    pts[2*i+1] = nucs[i].a2
tree = cKDTree(pts)
raw_pairs = tree.query_pairs(r=30.0)
cands_all = set()
for u,v in raw_pairs:
    ii,jj = u//2, v//2
    if ii != jj:
        cands_all.add(tuple(sorted((ii,jj))))

# Complement filter
cutoff_sq = 30.0**2
passed = 0
rejected = 0
for (i,j) in sorted(cands_all):
    d01 = np.sum((nucs[i].a1 - nucs[j].a2)**2)
    d10 = np.sum((nucs[i].a2 - nucs[j].a1)**2)
    if d01 < cutoff_sq or d10 < cutoff_sq:
        passed += 1
    else:
        rejected += 1

print(f'\nKDTree raw candidates (any arm pair): {len(cands_all)}')
print(f'Passed complement filter (arm0<->arm1 within 30nm): {passed}')
print(f'Rejected by complement filter: {rejected}')
if len(cands_all) > 0:
    print(f'Rejection rate: {rejected/len(cands_all)*100:.1f}%')
