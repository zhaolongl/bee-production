"""Six-node TAN, BDeu ESS=5, exact discrete inference without pgmpy.

The Chow-Liu tree maximizes conditional mutual information among the five
features given HazardCat. Item is the tree root and is ALWAYS marginalized
when evaluating category prediction. Numpy broadcasting gives the full exact
joint; marginalization is equivalent to variable elimination at these sizes.
No edge is interpreted causally. State vocabularies are fixed across folds.
"""
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from bee_common import *

NODES = ['HazardCat','Item','ProdR','SampR','Product','Stage']

class TAN:
    def __init__(self, states, parents=None, ess=ESS):
        self.states = states
        self.maps = {k:{s:j for j,s in enumerate(v)} for k,v in states.items()}
        self.parents, self.ess = parents, ess

    def encode(self,d):
        return np.column_stack([d[n].map(self.maps[n]).to_numpy(dtype=int) for n in NODES])

    def learn(self,d):
        # Conditional mutual information with zero cells contributing zero.
        x = self.encode(d); edges=[]; n=len(d)
        for a,b in itertools.combinations(range(1,6),2):
            counts=np.zeros((len(self.states[NODES[0]]),len(self.states[NODES[a]]),len(self.states[NODES[b]])))
            np.add.at(counts,(x[:,0],x[:,a],x[:,b]),1)
            nc=counts.sum((1,2),keepdims=True)
            den=counts.sum(2,keepdims=True)*counts.sum(1,keepdims=True)
            mask=counts>0
            ratio=np.divide(counts*nc,den,out=np.ones_like(counts),where=den>0)
            score=float((counts[mask]/n*np.log(ratio[mask])).sum())
            edges.append((score,a,b))
        # Deterministic maximum spanning tree (Kruskal), then orient from Item.
        roots=list(range(6)); adj={j:[] for j in range(1,6)}
        def root(j):
            while roots[j]!=j: j=roots[j]
            return j
        chosen=[]
        for score,a,b in sorted(edges,key=lambda z:(-z[0],z[1],z[2])):
            ra,rb=root(a),root(b)
            if ra!=rb:
                roots[ra]=rb;adj[a].append(b);adj[b].append(a);chosen.append((score,a,b))
        parents={'HazardCat':[], 'Item':['HazardCat']}; seen={1}; todo=[1]
        while todo:
            a=todo.pop(0)
            for b in sorted(adj[a]):
                if b not in seen:
                    seen.add(b);todo.append(b);parents[NODES[b]]=['HazardCat',NODES[a]]
        self.parents=parents
        self.cmi=chosen

    def counts(self,d,weights=None):
        x=self.encode(d); out={}
        for j,n in enumerate(NODES):
            names=[n]+self.parents[n]
            arr=np.zeros(tuple(len(self.states[k]) for k in names))
            np.add.at(arr,tuple(x[:,NODES.index(k)] for k in names),1 if weights is None else weights)
            out[n]=arr
        return out

    def fit_counts(self,counts):
        self.count=counts; self.cpds={}
        for n,arr in counts.items():
            smooth=np.clip(arr,0,None)+self.ess/arr.size
            self.cpds[n]=smooth/smooth.sum(axis=0,keepdims=True)
        joint=np.ones(tuple(len(self.states[n]) for n in NODES))
        for n,arr in self.cpds.items():
            names=[n]+self.parents[n]
            order=sorted(range(len(names)),key=lambda j:NODES.index(names[j]))
            axes=sorted(NODES.index(k) for k in names)
            shape=[1]*6
            for a in axes:shape[a]=len(self.states[NODES[a]])
            joint*=arr.transpose(order).reshape(shape)
        self.joint=joint/joint.sum()
        return self

    def fit(self,d,weights=None):
        if self.parents is None:self.learn(d)
        return self.fit_counts(self.counts(d,weights))

    def query(self,variables,evidence=None):
        evidence=evidence or {}
        sl=tuple(self.maps[n][evidence[n]] if n in evidence else slice(None) for n in NODES)
        arr=self.joint[sl]; remaining=[n for n in NODES if n not in evidence]
        marginal=arr.sum(axis=tuple(j for j,n in enumerate(remaining) if n not in variables))
        natural=[n for n in remaining if n in variables]
        if len(variables)>1:marginal=marginal.transpose([natural.index(n) for n in variables])
        return marginal/marginal.sum()

    def predict_proba(self,d):
        # marginalize hidden Item, leaving C x ProdR x SampR x Product x Stage
        probs=self.joint.sum(axis=1)
        idx=tuple(d[n].map(self.maps[n]).to_numpy(int) for n in NODES[2:])
        pred=probs[(slice(None),)+idx].T
        return pred/pred.sum(axis=1,keepdims=True)

def vocabulary(d):return {n:sorted(d[n].unique()) for n in NODES}

def evaluate(d,states,train,test,scale,split):
    tr,te=d.iloc[train],d.iloc[test]
    assert not set(tr.group_id)&set(te.group_id)
    model=TAN(states).fit(tr)
    pp=model.predict_proba(te); labels=np.array(states['HazardCat']); pred=labels[pp.argmax(1)]
    base=tr.HazardCat.mode().iloc[0]
    y=te.HazardCat
    yy=np.array([model.maps['HazardCat'][x] for x in y])
    metrics={'scale':scale,'split':split,'n_train':len(tr),'n_test':len(te),
       'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
       'macro_f1':f1_score(y,pred,labels=labels,average='macro',zero_division=0),
       'log_loss':log_loss(yy,pp,labels=np.arange(len(labels))),
       'majority_accuracy':float((y==base).mean()),'all_test_records_evaluated':True,
       'heldout_groups':te.group_id.nunique(),'training_groups':tr.group_id.nunique(),
       'brier_multiclass':np.mean(np.sum((pp-np.eye(len(labels))[yy])**2,axis=1))}
    predictions=te[['record_id','group_id','HazardCat']].copy()
    predictions['predicted']=pred;predictions['split']=split;predictions['scale']=scale
    for j,l in enumerate(labels):predictions['P_'+l]=pp[:,j]
    return metrics,predictions

def main():
    rows=[];predictions=[]
    for province in [False,True]:
        d=bn_data(province);states=vocabulary(d); scale='Province' if province else 'Region'
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.group_id))
        splits=[('Holdout 80:20',tr,te)]
        for k,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(d,d.HazardCat,d.group_id)):
            splits.append((f'CV{k+1}',tr,te))
        # Same groups are removed from early data if they recur in the temporal test.
        te=np.flatnonzero(d.year>=2024); testgroups=set(d.iloc[te].group_id)
        tr=np.flatnonzero((d.year<=2023)&~d.group_id.isin(testgroups))
        splits.append(('Temporal 2024-2026',tr,te))
        for name,tr,te in splits:
            m,p=evaluate(d,states,tr,te,scale,name);rows.append(m);predictions.append(p)
        print('BN validation finished:',scale,flush=True)
        if not province:
            full=TAN(states).fit(d)
            jsave({'states':states,'parents':full.parents,'ess':ESS,'nodes':NODES},'bn_protocol')
            np.savez_compressed(OUT/'bn_counts.npz',**full.count)
            save(d,'bn_input')
            cp=[]
            for region in states['SampR']:
                p=full.query(['Item'],{'SampR':region})
                for it,v in zip(states['Item'],p):cp.append({'sampling_region':region,'item':it,'probability':v})
            save(pd.DataFrame(cp),'bn_item_by_sampling_region')
            spatial=[]
            weights=d.HazardCat.value_counts()/NT
            for cat in states['HazardCat']:
                mat=full.query(['ProdR','SampR'],{'HazardCat':cat})
                for a,pr in enumerate(states['ProdR']):
                    for b,sr in enumerate(states['SampR']):spatial.append({'category':cat,'production_region':pr,'sampling_region':sr,'posterior':mat[a,b],'event_weight':weights[cat],'weighted_mass':mat[a,b]*weights[cat]})
            spatial=pd.DataFrame(spatial);spatial['footprint_share']=spatial.weighted_mass/spatial.weighted_mass.sum()
            save(spatial,'spatial_category_footprints')
            save(spatial.groupby(['production_region','sampling_region'],as_index=False).footprint_share.sum(),'spatial_footprint')
            layer=[]
            for item in states['Item']:
                for node in ['ProdR','Product','Stage']:
                    for s,p in zip(states[node],full.query([node],{'Item':item})):
                        layer.append({'item':item,'layer':node,'stratum':s,'probability':p})
                pr=full.query(['ProdR'],{'Item':item});unknown=sum(p for s,p in zip(states['ProdR'],pr) if s=='Other / unknown')
                for s,p in [('Location available',1-unknown),('Other / unknown',unknown)]:layer.append({'item':item,'layer':'Trace','stratum':s,'probability':p})
            save(pd.DataFrame(layer),'bn_stratum_probabilities')
            # Prior and equal-batch-weight sensitivities retain all events and categories.
            sensitivity=[]
            ref=full.query(['ProdR','SampR'])
            for ess in [1.,5.,20.]:
                model=TAN(states,full.parents,ess).fit(d)
                sensitivity.append({'analysis':'ESS','value':ess,'joint_L1_vs_primary':np.abs(model.query(['ProdR','SampR'])-ref).sum()})
            weights=1/d.groupby('record_id').record_id.transform('size').to_numpy()
            model=TAN(states,full.parents).fit(d,weights)
            sensitivity.append({'analysis':'Equal total weight per report','value':1,'joint_L1_vs_primary':np.abs(model.query(['ProdR','SampR'])-ref).sum()})
            save(pd.DataFrame(sensitivity),'bn_sensitivity')
    save(pd.DataFrame(rows),'bn_validation');save(pd.concat(predictions,ignore_index=True),'bn_predictions')

def load_model():
    p=json.loads((OUT/'bn_protocol.json').read_text('utf-8'))
    c=dict(np.load(OUT/'bn_counts.npz'))
    return TAN(p['states'],p['parents'],p['ess']).fit_counts(c)

if __name__=='__main__':main()
