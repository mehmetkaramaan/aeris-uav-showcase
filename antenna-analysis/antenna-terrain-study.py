import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json,math
from pathlib import Path
from IPython.display import display,Markdown
OUT=Path('/kaggle/working/antenna_results') if Path('/kaggle/working').exists() else Path('antenna_results')
OUT.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'figure.dpi':120,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#354c5e','#4c775f','#9c7840','#8c5860','#6c6190','#327f8b','#727234']
def figure(name):
    plt.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.show()


DATA_PATH = Path('flight_geometry_raw.csv')
DATA_URL = 'https://raw.githubusercontent.com/mehmetkaramaan/aeris-uav-showcase/main/antenna-analysis/flight_geometry_raw.csv'

if '__file__' in globals():
    DATA_PATH = Path(__file__).resolve().parent / DATA_PATH

if not DATA_PATH.exists():
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

flight_raw = pd.read_csv(DATA_PATH)
display(flight_raw.head())

required_columns = ['t', 'east', 'north', 'height', 'roll', 'pitch']
missing_columns = set(required_columns) - set(flight_raw.columns)
if missing_columns:
    raise ValueError(f'Missing columns: {sorted(missing_columns)}')

flight_raw = flight_raw[required_columns].copy()
for column in required_columns:
    flight_raw[column] = pd.to_numeric(flight_raw[column], errors='raise')

if not np.isfinite(flight_raw.to_numpy()).all():
    raise ValueError('Flight data contains missing or non-finite values')
if not flight_raw['t'].diff().iloc[1:].gt(0).all():
    raise ValueError('Flight timestamps must increase strictly')

display(flight_raw.describe().round(3))

airborne = flight_raw['height'] > 1.5
flight = flight_raw.loc[airborne, required_columns].copy()
flight = flight.round(6)
flight = flight.reset_index(drop=True)

if len(flight) < 2:
    raise ValueError('At least two airborne samples are needed')

r = {}
for column in required_columns:
    r[column] = flight[column].to_numpy(dtype=float)

flight.to_csv(OUT / 'flight_geometry.csv', index=False)
display(pd.DataFrame({
    'raw_samples': [len(flight_raw)],
    'airborne_samples': [len(flight)],
    'included_duration_s': [round(flight['t'].iloc[-1] - flight['t'].iloc[0], 2)]
}))


CONFIGS=[
 dict(id='compact',name='Compact omnidirectional',gain=0.,kind='omni',beam=100,cost=0,diversity=False),
 dict(id='dipole',name='Wide-angle dipole',gain=2.15,kind='omni',beam=78,cost=1,diversity=False),
 dict(id='collinear',name='High-gain collinear',gain=6.,kind='omni',beam=30,cost=2,diversity=False),
 dict(id='patch',name='Fixed directional panel',gain=8.,kind='directional',beam=70,cost=2,diversity=False),
 dict(id='yagi',name='Fixed Yagi',gain=12.,kind='directional',beam=40,cost=3,diversity=False),
 dict(id='tracker',name='Tracking directional panel',gain=8.,kind='tracker',beam=70,cost=4,diversity=False),
 dict(id='diversity',name='Omni with crossed UAV antennas',gain=2.15,kind='omni',beam=78,cost=2,diversity=True)]

def fspl(distance_m,mhz=868.):
 return 20*np.log10(np.maximum(distance_m,1.))+20*np.log10(mhz)-27.55

from itertools import product
PROFILES = [
 dict(id='local',name='Small training field',scale=1.,height=7.,sigma=2.,block=0.,foliage=0.,tilt=12.),
 dict(id='open',name='Large open terrain',scale=20.,height=35.,sigma=3.,block=0.,foliage=0.,tilt=25.),
 dict(id='forest',name='Vegetated terrain',scale=12.,height=25.,sigma=5.,block=0.,foliage=.04,tilt=30.),
 dict(id='urban',name='Built-up terrain',scale=12.,height=25.,sigma=4.,block=18.,foliage=0.,tilt=30.)]
ROUTES=['recorded','perimeter','out_and_back']
SELECTION_SEEDS=list(range(20));VALIDATION_SEEDS=list(range(100,120))
N=180                                                               
LOCAL_BOXES=[(11,6,19,12),(24.5,-13,31.5,-5),(35.5,5,38.5,11)]
def mission_geometry(profile,route_id):
    t=np.linspace(0,555,N);u=t/t[-1];scale=profile['scale']
    if route_id=='recorded':
        e=np.interp(t,np.linspace(0,555,len(r['t'])),r['east']);n=np.interp(t,np.linspace(0,555,len(r['t'])),r['north'])
        label='Transformed recorded route'
    elif route_id=='perimeter':
        e=21+20*np.cos(2*np.pi*u);n=22*np.sin(2*np.pi*u);label='Synthetic perimeter search'
    else:
        e=2+40*(1-abs(2*u-1));n=-15+8*np.sin(4*np.pi*u);label='Synthetic out-and-back route'
    z=profile['height']*(1+.25*np.sin(2*np.pi*u))
    roll=np.radians(profile['tilt']*np.sin(8*np.pi*u));pitch=np.radians(profile['tilt']*.6*np.sin(6*np.pi*u))
    return dict(t=t,e=e*scale,n=n*scale,z=z,roll=roll,pitch=pitch,label=label)
def sites(profile):
    s=profile['scale'];return [('south',21*s,-35*s),('west',-15*s,0.),('corner',-12*s,-32*s)]
def features(profile,route_id,site,mast):
    q=mission_geometry(profile,route_id);_,se,sn=site
    de=q['e']-se;dn=q['n']-sn;dz=q['z']-mast;horizontal=np.hypot(de,dn);distance=np.sqrt(horizontal**2+dz**2)
    az=np.unwrap(np.arctan2(dn,de));el=np.arctan2(dz,np.maximum(horizontal,.01))
                                                                                 
    target_az=math.atan2(-sn,21*profile['scale']-se)
    target_el=math.atan2(profile['height']-mast,math.hypot(21*profile['scale']-se,sn))
    occlusion=np.zeros(N);depth=np.zeros(N)
    for bx,by,bxx,byy in LOCAL_BOXES:
        s=profile['scale'];lo=np.zeros(N);hi=np.ones(N);valid=np.ones(N,dtype=bool)
        for origin,delta,mn,mx in [(se,de,bx*s,bxx*s),(sn,dn,by*s,byy*s)]:
            parallel=np.abs(delta)<1e-9;valid &= ~parallel|((origin>=mn)&(origin<=mx))
            safe=np.where(parallel,1.,delta);a=(mn-origin)/safe;b=(mx-origin)/safe
            lo=np.maximum(lo,np.where(parallel,-np.inf,np.minimum(a,b)));hi=np.minimum(hi,np.where(parallel,np.inf,np.maximum(a,b)))
        hit=valid&(hi>lo)&(hi>0)&(lo<1);mid=np.clip((lo+hi)/2,0,1)
                                                                                     
        height_at=mast+mid*dz;blocked=hit&(height_at<12)
        occlusion += blocked
        depth += np.where(blocked,np.maximum(0,hi-lo)*distance,0.)
    return dict(**q,distance=distance,az=az,el=el,target_az=target_az,target_el=target_el,occlusion=occlusion,depth=depth,mast=mast)
FEATURES={(p['id'],rt,site[0],mast):features(p,rt,site,mast) for p in PROFILES for rt in ROUTES for site in sites(p) for mast in [2.,8.]}


display(pd.DataFrame(CONFIGS));display(pd.DataFrame(PROFILES))


def metrics(margin,t,reserve=10.):
    dt=np.diff(t,append=t[-1]+np.median(np.diff(t)));bad=margin<0;run=longest=0.
    for b,d in zip(bad,dt):
        run=run+d if b else 0.;longest=max(longest,run)
    return dict(coverage=float(np.average(margin>=reserve,weights=dt)*100),outage=float(np.average(bad,weights=dt)*100),longest=longest,p05=float(np.percentile(margin,5)),reserve_gap=max(0.,10-float(np.sort(margin)[max(0,int(np.floor(.01*len(margin))))]),-float(np.min(margin))))
def robust_link(f,p,c,seed,ground_eirp=20.,uav_eirp=10.,mhz=868.,extra_loss=0.,tracker_delay=None):
    rng=np.random.default_rng(seed)
    sensitivity=rng.uniform(-100,-94);cable=rng.uniform(2,4)+extra_loss;gain_error=rng.uniform(-1,1);beam_scale=rng.uniform(.85,1.15);body=rng.uniform(0,4)
    delay=rng.uniform(.2,2.) if tracker_delay is None else tracker_delay
    z=rng.normal(size=N);fade=np.empty(N);fade[0]=z[0]
    dt=np.diff(f['t'],prepend=f['t'][0]);rho=np.exp(-dt/12.)
    for i in range(1,N):fade[i]=rho[i]*fade[i-1]+math.sqrt(1-rho[i]**2)*z[i]
    pointing=rng.normal(0,3,N);freeze_start=rng.uniform(100,400);freeze_duration=rng.uniform(2,8)
    beam=c['beam']*beam_scale;el=np.degrees(f['el'])
    if c['kind']=='omni':ground_loss=np.minimum(25,12*(el/beam)**2)
    else:
        if c['kind']=='tracker':
            past=np.maximum(0,f['t']-delay);freeze=(f['t']>=freeze_start)&(f['t']<freeze_start+freeze_duration);past=np.where(freeze,freeze_start-delay,past)
            aim_az=np.interp(past,f['t'],f['az']);aim_el=np.interp(past,f['t'],f['el'])
        else:aim_az=f['target_az'];aim_el=f['target_el']
        da=np.degrees((f['az']-aim_az+np.pi)%(2*np.pi)-np.pi)+pointing
        dl=np.degrees(f['el']-aim_el)+pointing
        ground_loss=np.minimum(30,12*(da/beam)**2+12*(dl/beam)**2)
                                                                            
    axis=np.stack([np.sin(f['pitch'])*np.cos(f['roll']),-np.sin(f['roll']),np.cos(f['pitch'])*np.cos(f['roll'])],axis=1)
    ray=np.stack([np.cos(f['el'])*np.cos(f['az']),np.cos(f['el'])*np.sin(f['az']),np.sin(f['el'])],axis=1)
    def axis_loss(a):return -10*np.log10(np.maximum(.01,1-np.sum(a*ray,axis=1)**2))
    uav_loss=axis_loss(axis)+body
    if c['diversity']:
        other=np.stack([np.cos(f['pitch']),np.zeros(N),-np.sin(f['pitch'])],axis=1)
        uav_loss=np.minimum(uav_loss,axis_loss(other)+body*.5)+1.
    ground_rx=c['gain']+gain_error-ground_loss;uav_rx=2.15+gain_error-uav_loss
    propagation=fspl(f['distance'],mhz)+cable+p['block']*f['occlusion']+p['foliage']*f['depth']+p['sigma']*fade
                                                                                   
    down=ground_eirp-ground_loss+uav_rx-propagation
    up=uav_eirp-uav_loss+ground_rx-propagation
    margin=np.minimum(up,down)-sensitivity
    return dict(**metrics(margin,f['t']),margin=margin,up=up,down=down)
def passed(a):return a['coverage']>=99 and a['outage']<=.1 and a['longest']<=3.


def evaluate_options(seeds,only=None):
    rows=[]
    for p in PROFILES:
        for c in CONFIGS:
            for site in sites(p):
                for mast in [2.,8.]:
                    key=(p['id'],c['id'],site[0],mast)
                    if only is not None and key not in only:continue
                    for seed in seeds:
                        vals=[robust_link(FEATURES[(p['id'],rt,site[0],mast)],p,c,seed) for rt in ROUTES]
                        rows.append(dict(profile=p['id'],antenna=c['id'],site=site[0],mast=mast,seed=seed,coverage=min(x['coverage'] for x in vals),outage=max(x['outage'] for x in vals),longest=max(x['longest'] for x in vals),p05=min(x['p05'] for x in vals),passed=all(passed(x) for x in vals),complexity=c['cost']+(1 if mast==8 else 0),reserve_gap=max(v['reserve_gap'] for v in vals)))
    return pd.DataFrame(rows)

selection=evaluate_options(SELECTION_SEEDS)
keys=['profile','antenna','site','mast','complexity']
summary=selection.groupby(keys,as_index=False).agg(pass_rate=('passed','mean'),coverage_worst=('coverage','min'),p05_worst=('p05','min'),outage_worst=('outage','max'))
layouts=[]
for _,g in summary.groupby(['profile','antenna']):
    eligible=g[g.pass_rate>=.95]
    best=eligible.sort_values(['complexity','p05_worst'],ascending=[True,False]).iloc[0] if len(eligible) else g.sort_values(['pass_rate','p05_worst','complexity'],ascending=[False,False,True]).iloc[0]
    layouts.append(best)
frozen=pd.DataFrame(layouts)
option_keys={(x.profile,x.antenna,x.site,x.mast) for x in frozen.itertuples()}
test=evaluate_options(VALIDATION_SEEDS,option_keys)
test_summary=test.groupby(['profile','antenna','site','mast'],as_index=False).agg(test_pass_rate=('passed','mean'),test_mean_coverage=('coverage','mean'),test_worst_coverage=('coverage','min'),test_worst_p05=('p05','min'),test_max_outage=('outage','max'))
comparison=frozen.merge(test_summary,on=['profile','antenna','site','mast'],validate='one_to_one')
comparison.to_csv(OUT/'terrain_comparison.csv',index=False)
selection.drop(columns=['reserve_gap']).to_csv(OUT/'selection_trials.csv',index=False)
test.drop(columns=['reserve_gap']).to_csv(OUT/'independent_tests.csv',index=False)
print('Selection link calculations:',len(selection)*3,'| Independent test link calculations:',len(test)*3)
display(comparison.round(3))


leaders=[]
for p in PROFILES:
    g=comparison[comparison.profile==p['id']]
    eligible=g[g.pass_rate>=.95]
    if len(eligible):row=eligible.sort_values(['complexity','p05_worst'],ascending=[True,False]).iloc[0]
    else:row=g.sort_values(['pass_rate','p05_worst','complexity'],ascending=[False,False,True]).iloc[0]
    c=next(x for x in CONFIGS if x['id']==row.antenna)
    leaders.append(dict(terrain=p['name'],profile=p['id'],antenna=c['name'],antenna_id=c['id'],station=row.site,mast_m=float(row.mast),selection_pass_rate=float(row.pass_rate),independent_pass_rate=float(row.test_pass_rate),mean_reserve_coverage=float(row.test_mean_coverage),worst_reserve_coverage=float(row.test_worst_coverage),worst_p05_margin_db=float(row.test_worst_p05),interpretation='Simplest configuration meeting the selection criterion' if len(eligible) else 'Strongest relative performer; not a universal suitability claim'))
leaders=pd.DataFrame(leaders);display(leaders.round(3))
leaders.to_csv(OUT/'terrain_findings.csv',index=False)
(OUT/'terrain_findings.json').write_text(json.dumps(leaders.to_dict('records'),indent=2,allow_nan=False))
fig,ax=plt.subplots(figsize=(11,5))
heat=comparison.pivot(index='antenna',columns='profile',values='test_pass_rate').reindex([c['id'] for c in CONFIGS])
im=ax.imshow(heat.values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
ax.set_yticks(range(len(heat)),[c['name'] for c in CONFIGS]);ax.set_xticks(range(len(heat.columns)),[next(p['name'] for p in PROFILES if p['id']==k) for k in heat.columns],rotation=15,ha='right')
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):ax.text(j,i,f'{100*heat.iloc[i,j]:.0f}%',ha='center',va='center',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
ax.set_title('Independent scenario pass rate at frozen station layouts');fig.colorbar(im,ax=ax,label='Empirical pass rate');plt.tight_layout();figure('antenna-terrain-comparison')


fig,axs=plt.subplots(2,2,figsize=(13,9))
for ax,p in zip(axs.flat,PROFILES):
    g=comparison[comparison.profile==p['id']].set_index('antenna').reindex([c['id'] for c in CONFIGS])
    ax.barh(range(7),g.test_mean_coverage,color=COLORS);ax.set_yticks(range(7),[c['name'] for c in CONFIGS],fontsize=7);ax.set(xlim=(0,101),xlabel='Mean reserve coverage (%)',title=p['name']);ax.axvline(99,color='black',ls='--',lw=.8)
plt.tight_layout();figure('antenna-terrain-coverage')
fig,axs=plt.subplots(2,2,figsize=(13,9))
for ax,p in zip(axs.flat,PROFILES):
    q=mission_geometry(p,'recorded');ax.plot(q['e'],q['n'],lw=1,label='Transformed recorded route')
    for site in sites(p):ax.scatter(site[1],site[2],marker='^',s=50,label=site[0])
    lead=leaders[leaders.profile==p['id']].iloc[0];site=next(x for x in sites(p) if x[0]==lead.station)
    ax.scatter(site[1],site[2],s=170,facecolors='none',edgecolors='black',label=f'{lead.antenna_id} / {lead.mast_m:g} m')
    for a,b,c,d in LOCAL_BOXES:
        s=p['scale'];ax.add_patch(plt.Rectangle((a*s,b*s),(c-a)*s,(d-b)*s,color='#8a9297',alpha=.3))
    ax.set(title=p['name'],xlabel='East (m)',ylabel='North (m)');ax.axis('equal');ax.legend(fontsize=6)
plt.tight_layout();figure('antenna-station-layouts')


assert set(SELECTION_SEEDS).isdisjoint(VALIDATION_SEEDS)
assert len(selection)==4*7*3*2*20 and len(test)==4*7*20
assert len(comparison)==28 and len(leaders)==4
assert abs(float(fspl(1000,2400))-100.054)<.02
for p in PROFILES:
    f=FEATURES[(p['id'],'recorded','south',2.)];a=robust_link(f,p,CONFIGS[1],3);b=robust_link(f,p,CONFIGS[1],3,extra_loss=6.)
    np.testing.assert_allclose(a['margin']-b['margin'],6.)
    assert np.isfinite(a['margin']).all()
print('Reproducibility and internal-consistency checks passed.')
