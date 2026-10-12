"""Fixed-protocol sufficient-statistic federation and central-DP simulations.

Privacy unit: one report contributing at most five distinct item rows.
Six tables -> add/remove L1 sensitivity <=30. Laplace noise is added ONCE to
the aggregated concatenated count vector, not at clients. Learned structure,
vocabulary, PC and HQ are public/fixed inputs in this conditional experiment;
the whole research pipeline is not claimed differentially private.
"""
import numpy as np
import pandas as pd
from bee_common import *
from bee_bn import TAN, NODES, load_model
from bee_optim import cells, inputs, allocate, metrics, MINIMUM

EPS=[.1,.25,.5,1.,2.,5.,10.];REPEATS=20;CAP=5

def pt_frame(model,uniform=False):
    mat=model.query(['Item','Product'])
    mat=mat/mat.sum(axis=1,keepdims=True)
    if uniform:mat[:]=1/mat.shape[1]
    return pd.DataFrame([{'item':it,'layer':'Product','stratum':s,'probability':mat[j,k]}
       for j,it in enumerate(model.states['Item']) for k,s in enumerate(model.states['Product'])])

def main():
    d=bn_data(); pooled=load_model();states=pooled.states;parents=pooled.parents
    assert d.groupby('record_id').size().max()<=CAP
    client_names=sorted(d.Client.unique()); ccounts={c:pooled.counts(d[d.Client.eq(c)]) for c in client_names}
    summed={n:sum(ccounts[c][n] for c in client_names) for n in NODES}
    fed=TAN(states,parents).fit_counts(summed)
    countgap=max(np.max(abs(summed[n]-pooled.count[n])) for n in NODES)
    cpdgap=max(np.max(abs(fed.cpds[n]-pooled.cpds[n])) for n in NODES)
    pc,hq,_=inputs();pt=pt_frame(pooled);a=cells('Product',pc,hq,pt)
    min_client=MINIMUM*len(a);minimum_total=min_client*len(client_names)
    budget=max(15000,int(np.ceil(minimum_total/5000)*5000))
    n0=allocate(a,budget);ref=metrics(a,n0);rows=[]
    def result(label,n,l1=0,eps=np.nan):
        return {'regime':label,'budget':budget,'epsilon':eps,**metrics(a,n),
            'loss_pp':(ref['coverage']-metrics(a,n)['coverage'])*100,
            'allocation_overlap':np.minimum(n0,n).sum()/budget,'pt_mean_L1':l1}
    rows.append(result('Centralized',n0));af=cells('Product',pc,hq,pt_frame(fed));nf=allocate(af,budget);rows.append(result('Federated counts',nf))
    extra=budget-minimum_total;sizes=np.array([(d.Client==c).sum() for c in client_names]);quota=extra*sizes/len(d)
    budgets=np.floor(quota).astype(int)+min_client;order=np.argsort(-(quota-np.floor(quota)),kind='stable');budgets[order[:budget-budgets.sum()]]+=1
    local=np.zeros(len(a),int);partitioned=np.zeros(len(a),int);clientrows=[];local_cells={}
    for c,bud,size in zip(client_names,budgets,sizes):
        lm=TAN(states,parents).fit_counts(ccounts[c]);lp=pt_frame(lm,uniform=size<20);al=cells('Product',pc,hq,lp)
        local+=allocate(al,int(bud));partitioned+=allocate(a,int(bud));local_cells[c]=al
        l1=np.abs(al.probability.to_numpy()-a.probability.to_numpy()).sum()/len(pc)
        clientrows.append({'client':c,'event_rows':size,'unique_reports':d.loc[d.Client.eq(c),'record_id'].nunique(),'budget':bud,
            'mode':'Local BDeu' if size>=20 else 'Uniform stratum prior','pt_mean_L1':l1,
            'synthetic_unresolved_client':c=='Unresolved location'})
    rows.append(result('Pooled parameters, split budgets',partitioned));rows.append(result('Local models, split budgets',local,np.average([r['pt_mean_L1'] for r in clientrows],weights=sizes)))
    save(pd.DataFrame(clientrows),'federated_clients')
    # Secondary policy: one national five-per-cell minimum, rather than 31 copies.
    # Remaining client budgets are split by event shares; within-client minima=0.
    policy=[]
    for btotal in [15000,30000]:
        rem=btotal-MINIMUM*len(a);qq=rem*sizes/len(d);bb=np.floor(qq).astype(int)
        oo=np.argsort(-(qq-np.floor(qq)),kind='stable');bb[oo[:rem-bb.sum()]]+=1
        npart=np.full(len(a),MINIMUM,int);nlocal=npart.copy()
        for c,bc in zip(client_names,bb):
            npart+=allocate(a,int(bc),minimum=0);nlocal+=allocate(local_cells[c],int(bc),minimum=0)
        nglobal=allocate(a,btotal);vglobal=metrics(a,nglobal)['coverage']
        for label,nn in [('Centralized',nglobal),('Pooled parameters, split budgets',npart),('Local models, split budgets',nlocal)]:
            policy.append({'budget':btotal,'regime':label,'minimum_policy':'One shared national minimum',**metrics(a,nn),'loss_pp':(vglobal-metrics(a,nn)['coverage'])*100})
    save(pd.DataFrame(policy),'federated_minimum_policy_sensitivity')
    rng=np.random.default_rng(SEED+81);runs=[]
    for eps in EPS:
        for rep in range(REPEATS):
            dp={n:np.maximum(0,summed[n]+rng.laplace(0,len(NODES)*CAP/eps,summed[n].shape)) for n in NODES}
            model=TAN(states,parents).fit_counts(dp);aa=cells('Product',pc,hq,pt_frame(model));nn=allocate(aa,budget)
            l1=np.abs(aa.probability.to_numpy()-a.probability.to_numpy()).sum()/len(pc)
            rr=result('Central-DP aggregated counts',nn,l1,eps);rr['replicate']=rep;rr['laplace_scale']=len(NODES)*CAP/eps;runs.append(rr)
        print('DP experiments finished: epsilon',eps,flush=True)
    dr=pd.DataFrame(runs);summary=dr.groupby('epsilon').agg(coverage_mean=('coverage','mean'),coverage_sd=('coverage','std'),
        loss_pp_mean=('loss_pp','mean'),loss_pp_sd=('loss_pp','std'),overlap_mean=('allocation_overlap','mean'),overlap_sd=('allocation_overlap','std'),pt_L1_mean=('pt_mean_L1','mean')).reset_index()
    for _,s in summary.iterrows():rows.append({'regime':'Central DP','epsilon':s.epsilon,'budget':budget,'coverage':s.coverage_mean,'loss_pp':s.loss_pp_mean,'allocation_overlap':s.overlap_mean,'pt_mean_L1':s.pt_L1_mean})
    save(pd.DataFrame(rows),'federated_comparison');save(dr,'federated_dp_runs');save(summary,'federated_dp_summary')
    values_per_client=sum(v.size for v in pooled.count.values());dense=values_per_client*len(client_names)*4
    nonzero=sum(np.count_nonzero(v) for cc in ccounts.values() for v in cc.values());sparse=nonzero*8
    raw=len(d)*len(NODES)*4
    comm=pd.DataFrame([{'payload':'Encoded raw event rows','values':len(d)*6,'bytes':raw,'definition':'six uint32 codes per event'},
                      {'payload':'Dense count upload, all clients','values':values_per_client*len(client_names),'bytes':dense,'definition':'uint32 count for every declared cell and client'},
                      {'payload':'Sparse count upload, all clients','values':nonzero*2,'bytes':sparse,'definition':'uint32 flattened index + uint32 count; ignores headers'},
                      {'payload':'One aggregated dense table set','values':values_per_client,'bytes':values_per_client*4,'definition':'server representation, not total communication'}])
    save(comm,'federated_communication')
    growth=[]
    for scale in [1,2,5,10,20,50,100,200,500,1000]:growth.append({'scale':scale,'event_rows':len(d)*scale,'raw_bytes':raw*scale,'dense_upload_bytes':dense,'raw_to_dense_ratio':raw*scale/dense})
    save(pd.DataFrame(growth),'federated_communication_scaling')
    jsave({'clients':len(client_names),'resolved_province_clients':len(client_names)-int('Unresolved location' in client_names),
          'budget':budget,'minimum_feasible_local_budget':minimum_total,'original_15000_feasible':minimum_total<=15000,
          'exact_count_max_gap':countgap,'exact_CPD_max_gap':cpdgap,'federated_allocation_identical':bool(np.array_equal(n0,nf)),
          'privacy_unit':'one reported record, maximum five distinct item rows','maximum_events_per_report':CAP,'count_L1_sensitivity':len(NODES)*CAP,
          'noise_location':'central aggregate','adjacency':'add/remove one capped report; client assignment and protocol fixed',
          'privacy_scope':'count-release experiment conditional on a public fixed protocol; not end-to-end private research',
          'repetitions':REPEATS,'seed':SEED+81,'loss_from_budget_partition_pp':rows[2]['loss_pp'],
          'additional_loss_local_parameters_pp':rows[3]['loss_pp']-rows[2]['loss_pp'],
          'raw_to_dense_ratio_observed':raw/dense,'raw_to_sparse_ratio_observed':raw/sparse},'federated_protocol')

if __name__=='__main__':main()
