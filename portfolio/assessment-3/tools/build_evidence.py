"""Render actual saved notebook cells for browser screenshots; does not rerun code."""
from pathlib import Path
import json,html,re
ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet.ipynb').read_text(encoding='utf-8'))
folder=ROOT/'output/evidence'
folder.mkdir(exist_ok=True)
ansi=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
for stem,index,title in [('architecture',11,'B.2 - Executed ANN architecture'),
                         ('results',16,'A.4 / B.4 - Executed comparison results'),
                         ('training',13,'B.3 - Executed training and checkpoint selection')]:
    cell=nb['cells'][index]
    body=[]
    if stem=='architecture':
        body.append('<h2>Executed code</h2><pre class="code">'+html.escape(''.join(cell['source']))+'</pre>')
    body.append('<h2>Stored notebook output</h2>')
    for output in cell['outputs']:
        if stem=='training' and output['output_type']=='stream': continue
        data=output.get('data',{})
        if 'text/html' in data: body.append(''.join(data['text/html']))
        elif 'text/plain' in data: body.append('<pre>'+html.escape(ansi.sub('',''.join(data['text/plain'])))+'</pre>')
        elif output['output_type']=='stream': body.append('<pre>'+html.escape(ansi.sub('',''.join(output['text'])))+'</pre>')
    page='''<!doctype html><meta charset="utf-8"><title>Portfolio 3 evidence</title>
    <style>body{font:15px Arial,sans-serif;background:#edf3f5;color:#213d4d;margin:0;padding:28px}
    main{background:white;max-width:1130px;padding:28px;margin:auto;border-radius:10px}
    h1{font-size:24px;margin:8px 0 16px}h2{font-size:16px;color:#087f8c;margin-top:22px}
    pre{font:13px Consolas,monospace;line-height:1.35;white-space:pre-wrap;background:#f5f8fa;padding:14px;overflow-wrap:anywhere}
    pre.code{font-size:12px;line-height:1.25}table{border-collapse:collapse;font-size:12px;width:100%}
    th,td{padding:7px 6px;text-align:right;border-bottom:1px solid #d6e2e7}th{background:#edf4f5}
    .tag{font-size:11px;color:#58717c;letter-spacing:.7px}.foot{font-size:12px;color:#607786;margin-top:18px}
    </style><main><div class="tag">COS40007 | PORTFOLIO 3 | AI-GENERATED REFERENCE</div>'''
    if stem in ['results','training']:
        hidden=7 if stem=='results' else 7
        page+=f'<style>main{{max-width:760px;margin:0}}table{{font-size:14px;table-layout:fixed}}th{{overflow-wrap:anywhere;font-size:12px}}td{{font-size:14px}}table tr>:nth-child(n+{hidden}){{display:none}}</style>'
        if stem=='training': page+='<style>th{font-size:11px}</style>'
    page+='<h1>'+title+'</h1><div>Nguyen Van Tuan Kiet | 104772099 | Cell execution '+str(cell['execution_count'])+'</div>'
    page+=''.join(body)+'<div class="foot">Rendered directly from saved executed notebook outputs. No values entered manually.</div></main>'
    (folder/(stem+'.html')).write_text(page,encoding='utf-8')
print('Prepared browser evidence pages from stored executed cells')
