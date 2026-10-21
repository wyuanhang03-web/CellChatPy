from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json
root=Path(__file__).resolve().parent
files={
'cheneyyu/pyccc':['docs/million_cell_scaling.md','examples/three_way_runtime_benchmark.py','src/pyccc/analysis.py','src/pyccc/expression.py'],
'Starlitnightly/omicverse':['omicverse/external/cellchat/_engine.py','omicverse/single/_cellchat.py','tests/single/test_cellchat.py'],
'RyanYip-Kat/PyCellChat':['src/pycellchat/modeling.py','compare_r/cellchat_ref.R'],
'saezlab/liana-py':['src/liana/method/sc/_cellchat.py']}
def fetch(task):
    repo,path=task
    d=root/repo.replace('/','__')
    meta=json.loads((d/'metadata.json').read_text())
    url='https://raw.githubusercontent.com/'+meta['full_name']+'/'+meta['default_branch']+'/'+path
    try:
        data=urllib.request.urlopen(url,timeout=35).read()
        p=d/'source'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        return {'url':url,'bytes':len(data)}
    except Exception as e:return {'url':url,'error':repr(e)}
with ThreadPoolExecutor(max_workers=4) as pool:
    print(json.dumps(list(pool.map(fetch,[(r,p) for r,paths in files.items() for p in paths])),indent=2))
