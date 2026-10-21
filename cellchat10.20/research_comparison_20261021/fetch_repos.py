from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,base64
ROOT=Path(__file__).resolve().parent
REPOS=['RyanYip-Kat/PyCellChat','cheneyyu/pyccc','saezlab/liana-py','Starlitnightly/omicverse','ventolab/CellphoneDB','zcang/COMMOT','earmingol/cell2cell']
def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CellChatPy-research'}),timeout=35))
def fetch(repo):
    out=ROOT/repo.replace('/','__');out.mkdir(exist_ok=True)
    try:
        m=get('https://api.github.com/repos/'+repo)
        (out/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
        readme=get('https://api.github.com/repos/'+repo+'/readme')
        content=base64.b64decode(readme['content']).decode('utf-8')
        (out/'README.md').write_text(content,encoding='utf-8')
        result={'repo':repo,'branch':m['default_branch'],'license':(m.get('license') or {}).get('spdx_id'),'description':m['description'],'readme_url':readme['html_url']}
        if repo in REPOS[:2]+['Starlitnightly/omicverse','saezlab/liana-py']:
            tree=get('https://api.github.com/repos/'+repo+'/git/trees/'+m['default_branch']+'?recursive=1')
            (out/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
            result['relevant_paths']=[x['path'] for x in tree['tree'] if x['type']=='blob' and any(t in x['path'].lower() for t in ['cellchat','benchmark','permut','quantile','parity'])]
        return result
    except Exception as e:return {'repo':repo,'error':repr(e)}
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,REPOS))
(ROOT/'index.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=True,indent=2))
