import json, pathlib, collections, concurrent.futures, cv2, numpy as np, hashlib
ROOT=pathlib.Path('/home/agilex/RL/robotcooking/files/data/session_102914')
OUT=pathlib.Path(__file__).resolve().parent
CAMS=('TOP_CAM','LEFT_WRIST_CAM','RIGHT_WRIST_CAM')
ARMS=('SLAVE_LEFT','SLAVE_RIGHT')

def stats(x):
 x=np.asarray(x,dtype=float)
 return {'min':float(x.min()),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95)),'p99':float(np.percentile(x,99)),'max':float(x.max())} if x.size else {}

def nearest(s,t):
 p=np.searchsorted(s,t); r=np.clip(p,0,len(s)-1); l=np.clip(p-1,0,len(s)-1)
 e1=np.abs(s[l]-t); e2=np.abs(s[r]-t); return np.where(e1<=e2,l,r),np.minimum(e1,e2)

def runs(mask):
 a=np.flatnonzero(np.diff(np.r_[False,mask,False])); return list(zip(a[::2].tolist(),a[1::2].tolist()))

def video(args):
 ep,cam,records=args
 cap=cv2.VideoCapture(str(ep/f'{cam}.mp4'))
 reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
 shape=[int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3]
 wanted={round((len(records)-1)*f):f for f in (0,.25,.5,.75,1)}
 n=dark=bright=dup=0; streak=max_streak=0; prev=None; means=[]
 while True:
  ok,frame=cap.read()
  if not ok: break
  small=cv2.resize(frame,(32,24)); mean=float(small.mean()); means.append(mean)
  dark+=mean<2; bright+=mean>253
  if prev is not None and np.array_equal(small,prev):
   dup+=1; streak+=1; max_streak=max(max_streak,streak)
  else: streak=0
  prev=small
  if n in wanted:
   im=cv2.resize(frame,(320,240)); cv2.putText(im,f'{ep.name[-6:]} {cam} {wanted[n]:.0%}',(7,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,255),1)
   cv2.imwrite(str(OUT/f'{ep.name}_{cam}_{wanted[n]:.2f}.jpg'),im)
  n+=1
 cap.release()
 return cam,{'reported_frames':reported,'decoded_frames':n,'fps':fps,'shape_hwc':shape,'matches_json':n==len(records),'dark_frames_mean_lt2':dark,'bright_frames_mean_gt253':bright,'thumbnail_equal_adjacent':dup,'longest_equal_thumbnail_sec':max_streak/fps,'brightness':stats(means)}

def audit(ep):
 d=json.loads((ep/'data.json').read_text()); m=json.loads((ep/'metadata.json').read_text())
 report={'id':ep.name,'metadata':m,'streams':{},'arms':{},'alignment':{}}
 ts={}
 for k,records in d.items():
  t=np.array([r['timestamp'] for r in records]); ts[k]=t; dt=np.diff(t)
  s={'count':len(t),'start':float(t[0]),'end':float(t[-1]),'duration':float(t[-1]-t[0]),'effective_hz':float((len(t)-1)/(t[-1]-t[0])),'dt_ms':stats(dt*1000),'nonincreasing':int((dt<=0).sum()),'nonfinite_timestamps':int((~np.isfinite(t)).sum()),'labels':dict(collections.Counter(str(r.get('label')) for r in records))}
  if k in CAMS: s['frame_indices_contiguous']=all(r['frame_index']==i for i,r in enumerate(records)); s['gaps_over_50ms']=int((dt>.05).sum())
  report['streams'][k]=s
  if k in ARMS:
   q=np.array([r['JOINT'] for r in records]); g=np.array([r['GRIPPER'] for r in records]); pose=np.array([r['POSE'] for r in records]); v=np.c_[q,g]
   if k=='SLAVE_LEFT':
    first_move=np.flatnonzero(np.any(np.abs(v-v[0])>.01,axis=1)); last_move=np.flatnonzero(np.any(np.abs(v-v[-1])>.01,axis=1))
    report['suggested_idle_review']={'onset_sec':float(t[first_move[0]]-t[0]) if first_move.size else None,'tail_sec':float(t[-1]-t[last_move[-1]]) if last_move.size else None,'threshold':'.01 rad or normalized gripper; review only, not trim command'}
   identical=np.all(np.diff(v,axis=0)==0,axis=1); rr=runs(identical)
   report['arms'][k]={'joint_shape':list(q.shape),'pose_shape':list(pose.shape),'nonfinite':int((~np.isfinite(v)).sum()+(~np.isfinite(pose)).sum()),'joint_min':q.min(0).tolist(),'joint_max':q.max(0).tolist(),'joint_span':np.ptp(q,axis=0).tolist(),'gripper_min':float(g.min()),'gripper_max':float(g.max()),'joint_max_step_rad':np.abs(np.diff(q,axis=0)).max(0).tolist(),'joint_velocity_abs_rad_s':stats(np.abs(np.diff(q,axis=0)/dt[:,None])),'gripper_max_step':float(np.abs(np.diff(g)).max()),'quaternion_norm':stats(np.linalg.norm(pose[:,3:],axis=1)),'exact_same_adjacent_fraction':float(identical.mean()),'max_identical_span_sec':max([float(t[b]-t[a]) for a,b in rr],default=0),'identical_spans_over_1s':[[round(float(t[a]-t[0]),3),round(float(t[b]-t[0]),3)] for a,b in rr if t[b]-t[a]>1], 'control_codes':dict(collections.Counter(str(r.get('IS_INTERVENTION')) for r in records))}
 t=ts['TOP_CAM']; aligns={}
 for k in (*CAMS,*ARMS):
  idx,err=nearest(ts[k],t); aligns[k]=(idx,err); report['alignment'][k]={'nearest_error_ms':stats(err*1000),'over_25ms':int((err>.025).sum()),'reused_samples':len(idx)-len(np.unique(idx))}
 for offset in (0,1,2):
  n=len(t)-offset; keep=np.ones(n,bool)
  for k,(idx,err) in aligns.items(): keep &= err[:n]<=.025
  future={}
  for k in ARMS:
   _,err=nearest(ts[k],t[offset:]); keep &= err<=.025; future[k]=stats(err*1000)
  selected=np.flatnonzero(keep); dropped=np.flatnonzero(~keep)
  report[f'conversion_offset_{offset}']={'candidate_frames':n,'kept_frames':int(keep.sum()),'dropped_frames':int((~keep).sum()),'dropped_indices':dropped.tolist(),'future_errors_ms':future,'retained_time_gaps_ms':stats(np.diff(t[selected])*1000),'keep_gaps_over_50ms':int((np.diff(t[selected])>.05).sum())}
 report['videos']=dict(video((ep,k,d[k])) for k in CAMS)
 return report

reports=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for r in ex.map(audit,sorted(ROOT.glob('episode_*'))):
  reports.append(r); print(r['id'],r['streams']['TOP_CAM']['count'],r['conversion_offset_2']['dropped_frames'],flush=True)
summary={'source':str(ROOT),'episodes':len(reports),'files':sum(1 for p in ROOT.rglob('*') if p.is_file()),'bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),'top_frames':sum(r['streams']['TOP_CAM']['count'] for r in reports),'top_duration_sec':sum(r['streams']['TOP_CAM']['duration'] for r in reports),'outcomes':dict(collections.Counter(r['metadata']['outcome'] for r in reports)),'instructions':dict(collections.Counter(r['metadata']['instruction'] for r in reports)),'decoded_frames':sum(v['decoded_frames'] for r in reports for v in r['videos'].values()),'video_mismatches':[[r['id'],k] for r in reports for k,v in r['videos'].items() if not v['matches_json']],'dropped_offset2':sum(r['conversion_offset_2']['dropped_frames'] for r in reports),'kept_offset2':sum(r['conversion_offset_2']['kept_frames'] for r in reports)}
(OUT/'audit.json').write_text(json.dumps({'summary':summary,'episodes':reports},ensure_ascii=False,indent=2))
for page in range(2):
 rows=[]
 for r in reports[page*10:(page+1)*10]:
  images=[cv2.imread(str(OUT/f"{r['id']}_TOP_CAM_{f:.2f}.jpg")) for f in (0,.25,.5,.75,1)]
  rows.append(np.hstack(images))
 if rows: cv2.imwrite(str(OUT/f'top_contact_{page+1}.jpg'),np.vstack(rows))
for idx in (0,9,18):
 r=reports[idx]; rows=[]
 for cam in CAMS:
  rows.append(np.hstack([cv2.imread(str(OUT/f"{r['id']}_{cam}_{f:.2f}.jpg")) for f in (0,.25,.5,.75,1)]))
 cv2.imwrite(str(OUT/f"three_cameras_{r['id'][-6:]}.jpg"),np.vstack(rows))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
