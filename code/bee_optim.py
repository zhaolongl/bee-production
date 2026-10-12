"""Seven allocation scenarios, integer budgets, stratum floors, Pareto mixtures and reserves.

One allocation unit is one batch-item assessment, NOT necessarily one unique
batch. All 46 items are retained. PC*Pt is a joint reported-event weight;
1-(1-p)^n is a model-defined coverage curve, not validated test sensitivity.
"""
import heapq
import numpy as np
import pandas as pd
from bee_common import *

BUDGETS=np.arange(5000,30001,250); REF=15000; MINIMUM=5
SCENARIOS={'S1':('Trace','All locations'),'S2':('Trace','Available locations'),
 'S3':('Trace','Other / unknown'),'S4':('Trace','Norfloxacin only'),
 'S5':('Product','Product type'),'S6':('Stage','Supply-chain stage'),'S7':('ProdR','Production region')}

def inputs():
    r,i,d=load_data()
    pc=d.groupby('item_en').record_id.nunique()/NT
    hq=pd.read_csv(OUT/'exposure.csv').groupby('item').hq_baseline.max().fillna(0)
    pt=pd.read_csv(OUT/'bn_stratum_probabilities.csv')
    return pc,hq,pt

def cells(layer,pc,hq,pt,scenario=None):
    a=pt[pt.layer.eq(layer)].copy()
    if scenario=='S2':a=a[a.stratum.eq('Location available')]
    if scenario=='S3':a=a[a.stratum.eq('Other / unknown')]
    if scenario=='S4':a=a[a.item.eq('Norfloxacin')]
    a=a.sort_values(['item','stratum']).reset_index(drop=True)
    a['pc']=a.item.map(pc);a['p']=a.pc*a.probability
    a['w_coverage']=a.p/pc.sum()
    a['hq_available']=a.item.map(hq).fillna(0).gt(0)
    a['w_health']=a.item.map(hq).fillna(0)*a.probability
    den=hq.sum()
    a['w_health_norm']=a.w_health/den if den>0 else 0
    return a

def allocate(a,budget,alpha=0.,floor=0.,reserve=None,minimum=MINIMUM):
    """Exact separable concave integer greedy; disjoint stratum floors OR one reserve.

    It is invalid to combine intersecting reserve and stratum floor constraints
    with this two-stage solver. Such combinations are explicitly rejected.
    """
    if floor and reserve is not None:raise ValueError('Intersecting constraints require a different solver')
    n=np.full(len(a),minimum,dtype=int)
    if n.sum()>budget:raise ValueError('Budget infeasible for five assessments per cell')
    p=a.p.to_numpy();w=(1-alpha)*a.w_coverage.to_numpy()+alpha*a.w_health_norm.to_numpy()
    def spend(idx,count):
        if count<=0:return
        if len(idx)==0:raise ValueError('Empty constrained subset')
        heap=[(-w[j]*p[j]*(1-p[j])**n[j],int(j)) for j in idx];heapq.heapify(heap)
        for _ in range(int(count)):
            gain,j=heapq.heappop(heap);n[j]+=1
            heapq.heappush(heap,(gain*(1-p[j]),j))
    if floor:
        strata=a.stratum.unique();quota=int(np.floor(budget*floor/len(strata)))
        need=sum(max(0,quota-n[a.stratum.eq(s)].sum()) for s in strata)
        if n.sum()+need>budget:raise ValueError('Budget infeasible for requested stratum floors')
        for s in strata:
            idx=np.flatnonzero(a.stratum.eq(s));spend(idx,max(0,quota-n[idx].sum()))
    if reserve is not None:
        idx,quota=reserve;need=max(0,quota-n[idx].sum())
        if n.sum()+need>budget:raise ValueError('Budget infeasible for requested reserve')
        spend(idx,need)
    spend(np.arange(len(a)),budget-n.sum())
    assert n.sum()==budget and n.min()>=minimum
    return n

def metrics(a,n):
    cover=-np.expm1(n*np.log1p(-a.p.to_numpy()))
    return {'coverage':float(cover@a.w_coverage),'hq_weighted_coverage':float(cover@a.w_health),
            'health_fraction':float(cover@a.w_health_norm)}

def baseline(a,budget,kind):
    n=np.full(len(a),MINIMUM,int);remaining=budget-n.sum()
    w=np.ones(len(a)) if kind=='Uniform' else a.p.to_numpy()
    quotas=remaining*w/w.sum();n+=np.floor(quotas).astype(int)
    order=np.argsort(-(quotas-np.floor(quotas)),kind='stable');n[order[:budget-n.sum()]]+=1
    return n

def main():
    pc,hq,pt=inputs();curves=[];allocations=[];contrib=[];knees=[];baselines=[]
    for scenario,(layer,label) in SCENARIOS.items():
        a=cells(layer,pc,hq,pt,scenario)
        local=[]
        for budget in BUDGETS:
            n=allocate(a,int(budget));m=metrics(a,n);row={'scenario':scenario,'layer':layer,'label':label,'budget':budget,**m};curves.append(row);local.append(row)
            vals=a[['item','stratum']].copy();vals['allocated']=n;vals['scenario']=scenario;vals['layer']=layer;vals['budget']=budget
            allocations.append(vals)
            cov=-np.expm1(n*np.log1p(-a.p.to_numpy()))*a.w_coverage
            cc=pd.DataFrame({'stratum':a.stratum,'contribution':cov}).groupby('stratum',as_index=False).sum();cc['scenario']=scenario;cc['budget']=budget;contrib.append(cc)
        t=pd.DataFrame(local);y=t.coverage.to_numpy();x=(BUDGETS-BUDGETS.min())/(BUDGETS.max()-BUDGETS.min())
        yn=(y-y[0])/(y[-1]-y[0]) if y[-1]>y[0] else y*0
        knee=int(np.argmax(yn-x));marginal=np.gradient(y,BUDGETS)*1000
        hit=np.flatnonzero(marginal<.01);reach=np.flatnonzero(y>=.95*y[-1])
        knees.append({'scenario':scenario,'layer':layer,'knee_budget':BUDGETS[knee],'coverage_at_knee':y[knee],
            'marginal_below_1pp_budget':BUDGETS[hit[0]] if len(hit) else np.nan,'reach_95pct_of_30000_budget':BUDGETS[reach[0]],
            'coverage_at_30000':y[-1],'relative_knee_coverage':y[knee]/y[-1]})
        for method in ['Uniform','Proportional','Optimized']:
            for budget in [5000,15000,30000]:
                n=allocate(a,budget) if method=='Optimized' else baseline(a,budget,method)
                baselines.append({'scenario':scenario,'layer':layer,'method':method,'budget':budget,**metrics(a,n)})
        print('Optimization finished:',scenario,flush=True)
    curve=pd.DataFrame(curves)
    curve['marginal_per_1000']=curve.groupby('scenario').coverage.transform(lambda y:np.gradient(y,250)*1000)
    save(curve,'optimization_curves');save(pd.concat(allocations),'optimization_allocations');save(pd.concat(contrib),'optimization_stratum_contributions')
    save(pd.DataFrame(knees),'optimization_knees');save(pd.DataFrame(baselines),'optimization_baselines')
    floors=[];flooralloc=[]
    for layer in ['Trace','Product','Stage','ProdR']:
        a=cells(layer,pc,hq,pt);base=metrics(a,allocate(a,REF))['coverage']
        for f in np.linspace(0,1,11):
            n=allocate(a,REF,floor=f);s=pd.Series(n).groupby(a.stratum).sum()
            floors.append({'layer':layer,'floor':f,'budget':REF,'min_per_stratum':s.min(),'max_per_stratum':s.max(),'effective_strata':int((s>=.01*REF).sum()),'n_strata':len(s),'loss_pp':(base-metrics(a,n)['coverage'])*100,**metrics(a,n)})
            if f in [0,.2,.5,1.]:
                v=a[['item','stratum']].copy();v['allocated']=n;v['layer']=layer;v['floor']=f;flooralloc.append(v)
    save(pd.DataFrame(floors),'optimization_floors');save(pd.concat(flooralloc),'optimization_floor_allocations')
    a=cells('Product',pc,hq,pt);pareto=[];pa=[]
    for alpha in np.linspace(0,1,21):
        n=allocate(a,REF,alpha=alpha);pareto.append({'alpha':alpha,'budget':REF,**metrics(a,n)})
        v=a[['item','stratum']].copy();v['allocated']=n;v['alpha']=alpha;pa.append(v)
    save(pd.DataFrame(pareto),'optimization_pareto');save(pd.concat(pa),'optimization_pareto_allocations')
    top=hq.sort_values(ascending=False).head(3).index;ix=np.flatnonzero(a.item.isin(top));res=[]
    for share in np.arange(0,.81,.1):
        n=allocate(a,REF,reserve=(ix,int(REF*share)))
        res.append({'reserved_share':share,'actual_share':n[ix].sum()/REF,'items':'; '.join(top),'budget':REF,**metrics(a,n)})
    save(pd.DataFrame(res),'optimization_health_reserves')
    # Denominator-supported alternative for product strata: use report counts / product total.
    r,i,d=load_data();tot=pd.read_csv(DATA/'product_denominators.csv').set_index('product_en').total
    q=a.copy();counts=d.groupby(['item_en','product_en']).record_id.nunique()
    q['p']=[counts.get((it,pr),0)/tot[pr] for it,pr in zip(q.item,q.stratum)]
    # Smoothed BN event weights remain fixed; this comparison isolates the response-curve input.
    n0=allocate(a,REF);n1=allocate(q,REF)
    save(pd.DataFrame([{'input':'Primary joint event weight',**metrics(a,n0)},
                      {'input':'Product report fraction response, own objective',**metrics(q,n1)},
                      {'input':'Product report fraction allocation evaluated on primary',**metrics(a,n1)}]),'optimization_product_denominator_sensitivity')
    save(cells('Trace',pc,hq,pt),'optimization_inputs_trace')
    jsave({'budget_min':5000,'budget_max':30000,'step':250,'points':len(BUDGETS),'scenarios':len(SCENARIOS),'all_hazards':len(pc),
           'minimum_per_cell':MINIMUM,'reference_budget':REF,'unit':'one batch-item assessment',
           'health_objective_scope':'Only ADI-supported screening subset; absent HQ is not zero health risk',
           'objective':'historical reported-event footprint coverage simulation; not realized detection or disease reduction'},'optimization_protocol')

if __name__=='__main__':main()
