"""Q12: recompute Section 4.5 regression / (partial|semi-partial) correlations from the persisted 16 per-row records.
Source: experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-reward-gradient-linkage/per_token_gradient_diagnostic_result_v2.json
Pure numpy; no scipy."""
import json, math
import numpy as np
P='/Users/researcher/Desktop/AISI/experiments/09d_sft_seeded_adversarial_rl/aws_runs/stage9d-reward-gradient-linkage/per_token_gradient_diagnostic_result_v2.json'
d=json.load(open(P))
rows=[r for s in d['step_analyses'] for r in s['per_row_details']]
y=np.array([r['total_abs_grad'] for r in rows]); a=np.abs([r['advantage'] for r in rows]); z=np.array([r['n_code_tokens'] for r in rows],float)
n=len(y); print('n rows',n, 'steps', [s['step'] for s in d['step_analyses']])
def ols(X,y):
    X=np.column_stack([np.ones(len(y))]+X); b,res,rk,sv=np.linalg.lstsq(X,y,rcond=None)
    e=y-X@b; df=len(y)-X.shape[1]; s2=e@e/df; cov=s2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
    r2=1-e@e/((y-y.mean())@(y-y.mean())); return b,se,b/se,df,r2,e
b,se,t,df,r2,e_full=ols([a,z],y)
print('FULL MODEL y ~ 1 + |adv| + n_code: coef',b,'se',se,'t',t,'df',df,'R2',r2)
print('corr(n_code,y)=%.4f corr(|adv|,y)=%.4f corr(|adv|,n_code)=%.4f'%(np.corrcoef(z,y)[0,1],np.corrcoef(a,y)[0,1],np.corrcoef(a,z)[0,1]))
def resid(v,x):
    X=np.column_stack([np.ones(n),x]); return v-X@np.linalg.lstsq(X,v,rcond=None)[0]
ry=resid(y,a); rz=resid(z,a)
semi=np.corrcoef(ry,z)[0,1]     # y residualised only (what the draft's wording describes)
partial=np.corrcoef(ry,rz)[0,1] # both residualised (true partial correlation)
print('semi-partial-type corr(resid(y|adv), n_code) = %.4f'%semi)
print('TRUE partial corr(resid(y|adv), resid(n_code|adv)) = %.4f'%partial)
for name,r in (('semi',semi),('partial',partial),('draft -0.21',-0.21)):
    print('t from r with 13 df (t=r*sqrt(13/(1-r^2))): %s -> %.4f'%(name, r*math.sqrt(13/(1-r*r))))
print('check identity: partial-corr t should equal coefficient t (%.4f): %.4f'%(t[2], partial*math.sqrt(13/(1-partial**2))))
# also naive correlation t if someone treats residual corr as ordinary Pearson with n-2=14 df
print('Pearson-type t with n-2=14 df for semi: %.4f'%(semi*math.sqrt(14/(1-semi**2))))
print('per-row table (n_code, |adv|, total_abs_grad):'); 
for r in rows: print(r['n_code_tokens'], round(abs(r['advantage']),6), r['total_abs_grad'])
