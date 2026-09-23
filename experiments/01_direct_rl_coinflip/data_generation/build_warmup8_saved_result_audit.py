"""Build a no-model audit of the saved warmup8 dry-run result."""
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').is_dir())  # repo root, robust to relocation
TARGET=ROOT/'notebooks'/'warmup8_saved_result_readonly_audit.ipynb'

def cell(kind,source):
    out={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
    if kind=='code': out.update({'execution_count':None,'outputs':[]})
    return out

cells=[cell('markdown','''# Warmup8 saved-result read-only audit

Reads the persisted warmup dry-run JSON from Drive. No package installation, GPU, model loading, generation, or training.
'''),cell('code',r'''import json,re,statistics
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive',force_remount=False)
root=Path('/content/drive/MyDrive/AISI/checkpoints')
logs=[]
for directory in root.glob('grpo-checkpoint500-corrected-fullreward-warmup8-v*'):
    path=directory/'checkpoint500_fullreward_warmup8_dryrun.json'
    if path.is_file() and path.stat().st_size: logs.append(path)
if not logs: raise RuntimeError('No saved warmup8 evidence log found.')
LOG=max(logs,key=lambda x:x.stat().st_mtime)
data=json.loads(LOG.read_text())
print({'log':str(LOG),'gate':data.get('gate'),'groups':len(data.get('groups',[])),
       'telemetry':len(data.get('telemetry',[]))})
'''),cell('code',r'''print('===== POST-UPDATE HELD-OUT COMPLETIONS =====')
post=data.get('post_rows',[])
if len(post)!=25: raise RuntimeError(f'Expected 25 saved post rows, found {len(post)}')
for i,row in enumerate(post,1):
    text=str(row.get('text','')); score=row.get('score',{})
    print('\n'+'='*100)
    print({'index':i,'truth':row.get('truth'),'r_task':score.get('r_task'),
           'p_structure':score.get('p_structure'),'p_state_variation':score.get('p_state_variation'),
           'generated_tokens':row.get('generated_tokens'),'ends_at_answer':
           text.rstrip().casefold().endswith('</answer>')})
    print(text)

summary={'post_rows':len(post),
 'valid_answer_format':sum(r.get('score',{}).get('r_task')!=-5.0 for r in post),
 'correct':sum(r.get('score',{}).get('r_task')==4.0 for r in post),
 'strict_structure':sum(r.get('score',{}).get('p_structure')==0.0 for r in post),
 'ended_at_answer':sum(str(r.get('text','')).rstrip().casefold().endswith('</answer>') for r in post),
 'empty_or_whitespace':sum(not str(r.get('text','')).strip() for r in post)}
print('\n===== POST SUMMARY ====='); print(json.dumps(summary,indent=2))
'''),cell('code',r'''print('===== ACCEPTED GROUP TREND =====')
accepted=[g for g in data.get('groups',[]) if g.get('accepted')]
for i,g in enumerate(accepted,1):
    rollouts=g.get('rollouts',[])
    print({'update':i,'attempt':g.get('attempt'),'fallback':g.get('fallback'),
      'accuracy':statistics.fmean(r['breakdown']['r_task']==4.0 for r in rollouts),
      'strict_structure':statistics.fmean(r['breakdown']['p_structure']==0.0 for r in rollouts),
      'variation_pass':statistics.fmean(r['breakdown']['p_state_variation']==0.0 for r in rollouts),
      'ended_at_answer':statistics.fmean(str(r['completion']).rstrip().casefold().endswith('</answer>') for r in rollouts),
      'reward_std':g.get('reward_std')})
print('READ-ONLY AUDIT COMPLETE. No model was loaded.')
''')]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
import sys as _sys; _sys.path.insert(0, str(ROOT))
from src.notebook_io import safe_write_notebook
safe_write_notebook(nb, TARGET); print(TARGET)
