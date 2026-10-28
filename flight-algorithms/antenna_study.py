import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent
CONFIGS = [dict(id='compact', name='Compact low-gain omni', gain=0.0, kind='omni', beam=100, cost=0, diversity=False), dict(id='dipole', name='Dipole / wide-angle omni', gain=2.15, kind='omni', beam=78, cost=1, diversity=False), dict(id='collinear', name='High-gain collinear', gain=6.0, kind='omni', beam=30, cost=2, diversity=False), dict(id='patch', name='Fixed directional panel', gain=8.0, kind='directional', beam=70, cost=2, diversity=False), dict(id='yagi', name='Fixed directional Yagi', gain=12.0, kind='directional', beam=40, cost=3, diversity=False), dict(id='tracker', name='Tracking directional panel', gain=8.0, kind='tracker', beam=70, cost=4, diversity=False), dict(id='diversity', name='Omni + crossed UAV antenna selection', gain=2.15, kind='omni', beam=78, cost=2, diversity=True)]
ENVS = [dict(id='open', name='Open terrain', obstruction=0.0, sigma=2.0), dict(id='buildings', name='Building shadowing', obstruction=15.0, sigma=4.0), dict(id='foliage', name='Vegetation / shadowing', obstruction=8.0, sigma=5.0)]
SOURCES = [dict(title='ITU-R P.525: free-space attenuation', url='https://www.itu.int/rec/R-REC-P.525-5-202411-I/en'), dict(title='ArduPilot: directional antenna tracking', url='https://ardupilot.org/copter/docs/common-antenna-tracking.html'), dict(title='Taoglas: 868 MHz collinear example; catalog gain is not the full radiation pattern', url='https://www.taoglas.com/datasheets/OMB.868.B10F21.pdf'), dict(title='L-com: 6 dBi omni with narrow vertical coverage', url='https://www.l-com.com/Images/Downloadables/Datasheets/ds_HGV-906U.pdf')]

def fspl(distance_m, mhz=868.0):
    return 20 * np.log10(np.maximum(distance_m, 1.0)) + 20 * np.log10(mhz) - 27.55

def pattern(c, az, el):
    if c['kind'] == 'omni':
        return c['gain'] - np.minimum(25.0, 12 * (el / c['beam']) ** 2)
    if c['kind'] == 'tracker':
        error = np.full_like(az, 5.0)
    else:
        error = (az + 180) % 360 - 180
    return c['gain'] - np.minimum(30.0, 12 * (error / c['beam']) ** 2 + 12 * (el / c['beam']) ** 2)

def route():
    data = json.loads((ROOT / 'validation/full-survey-v07.json').read_text())['records']
    r = [x for x in data if x['telemetry'].get('height', 0) > 1.5]
    return {k: np.array([float(x['elapsed'] if k == 't' else x['telemetry'].get(k, 0)) for x in r]) for k in ['t', 'east', 'north', 'height', 'roll', 'pitch']}

def geometry(r, scale=1.0):
    e = r['east'] * scale
    n = r['north'] * scale
    up = r['height'] - 2.0
    d = np.sqrt(e * e + n * n + up * up)
    az = np.degrees(np.arctan2(n, e))
    el = np.degrees(np.arctan2(abs(up), np.maximum(np.hypot(e, n), 0.1)))
    shadow = (e > 10 * scale) & (n > 2 * scale) | (e > 23 * scale) & (n < -6 * scale)
    return (d, az, el, shadow)

def evaluate(r, c, env, scale=1.0, eirp=20.0, sensitivity=-100.0, reserve=10.0, seed=0, mhz=868.0):
    d, az, el, shadow = geometry(r, scale)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=len(d))
    fade = np.empty(len(d))
    fade[0] = z[0]
    for i in range(1, len(d)):
        fade[i] = 0.95 * fade[i - 1] + math.sqrt(1 - 0.95 ** 2) * z[i]
    ground = pattern(c, az, el)
    tilt = np.degrees(np.hypot(r['roll'], r['pitch']))
    pose = np.minimum(20.0, 12 * (tilt / 60) ** 2 + 12 * (el / 78) ** 2)
    if c['diversity']:
        pose = np.minimum(pose, 3.0 + 12 * ((90 - el) / 78) ** 2) + 1.0
    uav = 2.15 - pose
    ground_tx = eirp - c['gain'] + ground
    uav_tx = eirp - 2.15 + uav
    loss = fspl(d, mhz) + 2.0 + env['obstruction'] * shadow + env['sigma'] * fade
    received = np.minimum(ground_tx + uav, uav_tx + ground) - loss
    margin = received - sensitivity
    dt = np.diff(r['t'], append=r['t'][-1] + np.median(np.diff(r['t'])))
    usable = margin >= reserve
    out = margin < 0
    longest = run = 0.0
    for bad, t in zip(out, dt):
        run = run + t if bad else 0.0
        longest = max(longest, run)
    return dict(coverage=float(np.average(usable, weights=dt) * 100), outage=float(np.average(out, weights=dt) * 100), p05=float(np.percentile(margin, 5)), longest=float(longest), margin=margin)

def recommend(rows):
    eligible = [x for x in rows if x['coverage_min'] >= 99.0 and x['outage_max'] <= 0.1]
    if not eligible:
        return None
    return min(eligible, key=lambda x: (x['cost'], -x['p05_mean']))['id']

def build():
    r = route()
    results = []
    for scale in [1.0, 20.0]:
        for env in ENVS:
            rows = []
            for c in CONFIGS:
                runs = [evaluate(r, c, env, scale, seed=s) for s in range(10)]
                rows.append(dict(**c, coverage_mean=float(np.mean([a['coverage'] for a in runs])), coverage_min=min((a['coverage'] for a in runs)), outage_max=max((a['outage'] for a in runs)), p05_mean=float(np.mean([a['p05'] for a in runs])), longest_max=max((a['longest'] for a in runs)), trace=[round(float(v), 2) for v in runs[0]['margin'][::4]]))
            results.append(dict(scale=scale, environment=env['id'], rows=rows, recommended=recommend(rows)))
    out = dict(configs=CONFIGS, environments=ENVS, results=results, times=r['t'][::4].round(2).tolist(), sources=SOURCES, assumptions=dict(frequency_mhz=868, peak_eirp_dbm=20, sensitivity_dbm=-100, reserve_db=10, cable_total_db=2, seeds=list(range(10)), station_height_m=2, route='Recorded v0.7 flight; only height >1.5 m; time-weighted percentages', selection='All 10 seeds: reserve coverage >=99%, outage <=0.1%; lowest complexity, then 5th percentile margin', scope='Telemetry link only. No real radio, measured antenna pattern, video bandwidth, packet delay or flight failsafe injection.'))
    (ROOT / 'panel/antenna-results.json').write_text(json.dumps(out, ensure_ascii=False))
    (ROOT / 'validation/antenna-study.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps([dict(scale=x['scale'], env=x['environment'], recommended=x['recommended']) for x in results], indent=2))
    return out
if __name__ == '__main__':
    build()
