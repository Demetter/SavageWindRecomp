"""Within-launch diagnostic survey analysis; never a throughput/promotion report."""
import argparse, json, re, statistics
from pathlib import Path

def percentile(values, q):
    v=sorted(values); x=(len(v)-1)*q; i=int(x)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)

def analyze(path):
    text=path.read_text(errors='replace')
    frames=[(int(t),int(f),int(b),int(e)) for t,f,b,e in re.findall(r"\[survey-frame\] (\d+) (\d+) (-?\d+) (\d+)",text)]
    first=next((t-b*10_000_000_000 for t,f,b,e in frames if b>=0),None)
    if first is None: return {'error':'No measurement blocks yet'}
    block_values={}; selected={}
    complete=int((frames[-1][0]-first)/10_000_000_000)
    for prev,cur in zip(frames,frames[1:]):
        t,f,b,e=cur; offset=(t-first)-b*10_000_000_000
        if b<0 or b>=complete or prev[2]!=b or not 1_000_000_000<=offset<9_000_000_000: continue
        dt=(t-prev[0])/1e6
        block_values.setdefault(b,[]).append(dt); selected[f]=b
    blocks=[]
    for b,v in sorted(block_values.items()):
        blocks.append({'block':b,'enabled':b%4 in (1,2),'frames':len(v),'mean_ms':statistics.mean(v),
                       'fps':1000/statistics.mean(v),'p99_ms':percentile(v,.99),'max_ms':max(v),'over_10ms':sum(x>10 for x in v)})
    cycles=[]
    for start in range(0,complete-3,4):
        if not all(b in block_values for b in range(start,start+4)): continue
        off=statistics.mean(statistics.mean(block_values[b]) for b in (start,start+3))
        on=statistics.mean(statistics.mean(block_values[b]) for b in (start+1,start+2))
        cycles.append({'cycle':start//4,'off_ms':off,'on_ms':on,'delta_ms':on-off,'sensitivity_nominal':(on-off)/.5})
    doses=[{'role':r,'id':i,'block':int(b),'count':int(c),'mean_ms':int(ns)/int(c)/1e6 if int(c) else 0,'max_ms':int(mx)/1e6}
           for r,i,b,c,ns,mx in re.findall(r"\[survey-dose\] role (\S+) id (\S+) block (\d+) count (\d+) ns (\d+) max_ns (\d+)",text)]
    gpu=[int(ns)/1e6 for f,b,ns,it in re.findall(r"\[survey-gpu-dose\] frame (\d+) block (-?\d+) ns (\d+) iterations (\d+)",text) if int(f) in selected and int(b)%4 in (1,2)]
    result={'run':path.parent.name,'blocks':blocks,'cycles':cycles,'cpu_doses':doses}
    if cycles:
        result.update(off_ms=statistics.mean(c['off_ms'] for c in cycles),on_ms=statistics.mean(c['on_ms'] for c in cycles),delta_ms=statistics.mean(c['delta_ms'] for c in cycles))
        result['off_fps']=1000/result['off_ms'];result['on_fps']=1000/result['on_ms']
        result['sensitivity_nominal']=result['delta_ms']/.5
    if gpu: result['gpu_dose']={'frames':len(gpu),'mean_ms':statistics.mean(gpu),'median_ms':statistics.median(gpu),'p99_ms':percentile(gpu,.99),'max_ms':max(gpu)}
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('log',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    result=analyze(a.log);s=json.dumps(result,indent=2)
    if a.out: a.out.write_text(s+'\n')
    print(s)
