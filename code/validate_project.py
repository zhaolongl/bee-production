"""Numerical and artifact checks on the data tables, results and figures."""
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from PIL import Image
from bee_common import *
from bee_bn import TAN, NODES, load_model
from bee_optim import allocate, metrics, inputs, cells
from bee_risk import edi, tox_score

def main():
    checks=[]
    def check(name,ok,detail=''):
        checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})
        if not ok:print('FAILED:',name,detail)
    r,i,d=load_data();check('reports_1798',len(r)==1798);check('raw_entries_1922',len(i)==1922);check('unique_events_1920',len(d)==1920)
    check('all_46_items_retained',d.item_en.nunique()==46);check('four_products_retained',set(d.product_en)==set(PRODUCTS))
    p=pd.read_csv(DATA/'product_denominators.csv');check('product_totals_match',p.total.sum()==NT and p.failed.sum()==len(r))
    # Only where the raw platform exports are available: they must be the files the tables were built from.
    for x in json.loads((DATA/'source_manifest.json').read_text('utf-8'))['inputs']:
        raw=RAW/x['file']
        if raw.is_file():check('raw_hash_unchanged_'+x['file'],hashlib.sha256(raw.read_bytes()).hexdigest()==x['sha256'])
    a=pd.read_csv(OUT/'product_associations.csv');check('odds_ratios_three_products_two_windows',len(a)==6 and (a.ci_low<a.odds_ratio).all() and (a.odds_ratio<a.ci_high).all())
    m=load_model();check('joint_normalization',np.isclose(m.joint.sum(),1));check('joint_positive',(m.joint>0).all())
    for n in NODES:check('CPD_normalization_'+n,np.allclose(m.cpds[n].sum(axis=0),1))
    # Independent one-state-context fixture: smoothed class posterior has known count ratio.
    tiny=pd.DataFrame({'HazardCat':['A','A','A','B'],'Item':['i']*4,'ProdR':['r']*4,'SampR':['s']*4,'Product':['p']*4,'Stage':['t']*4})
    states={n:sorted(tiny[n].unique()) for n in NODES};mm=TAN(states).fit(tiny)
    expected=np.array([(3+2.5)/9,(1+2.5)/9])
    check('TAN_exact_posterior_fixture',np.allclose(mm.predict_proba(tiny)[0],expected))
    query=m.query(['ProdR','SampR'],{'HazardCat':'Veterinary drugs'})
    check('conditional_joint_normalization',np.isclose(query.sum(),1))
    b=pd.read_csv(OUT/'bn_validation.csv');check('all_14_validation_experiments',len(b)==14);check('no_test_state_dropped',b.all_test_records_evaluated.all())
    pred=pd.read_csv(OUT/'bn_predictions.csv');check('predictions_match_validation_rows',len(pred)==b.n_test.sum())
    check('EDI_unit_fixture',np.isclose(edi(1000,10,60),1/6))
    check('ADI_score_monotone',np.all(np.diff([tox_score(x,'Pesticides') for x in np.geomspace(1e-8,100,10000)])<=1e-12))
    e=pd.read_csv(OUT/'exposure.csv');check('quantified_entries_703',e.n.sum()==703);check('9_ADI_chemicals',e.loc[e.adi_mgkg_bw_day.notna(),'item'].nunique()==9)
    for drug in ['Chloramphenicol','Norfloxacin','Nitrofurazone metabolite (SEM)','Lead']:
        check('no_ADI_assigned_'+drug,e.loc[e.item.eq(drug),'hq_baseline'].isna().all())
    critical=pd.read_csv(OUT/'critical_intake.csv')
    check('critical_intake_solves_HQ_one',np.allclose(edi(critical.concentration_ugkg,critical.critical_intake_g_day,critical.body_weight_kg)/(1000*critical.adi_mgkg_bw_day),1))
    check('critical_intake_all_summaries',set(critical.concentration_basis)=={'mean','median','maximum'} and set(critical.body_weight_kg)=={45.,60.,75.,90.})
    check('473_weights',len(pd.read_csv(OUT/'risk_weight_grid.csv'))==473)
    # Exact integer optimization independently checked by exhaustive enumeration.
    a=pd.DataFrame({'p':[.04,.09,.02],'w_coverage':[.3,.5,.2],'w_health_norm':[.2,.3,.5],'w_health':[.2,.3,.5],'stratum':['A','A','B']})
    for floor in [0,.6]:
        n=allocate(a,12,floor=floor,minimum=1);best=-1
        for n1 in range(1,11):
            for n2 in range(1,12-n1):
                nn=np.array([n1,n2,12-n1-n2])
                if floor and (nn[:2].sum()<3 or nn[2]<3):continue
                best=max(best,metrics(a,nn)['coverage'])
        check('integer_optimum_exhaustive_floor_'+str(floor),np.isclose(metrics(a,n)['coverage'],best))
    try:allocate(a,2,minimum=1);infeasible=False
    except ValueError:infeasible=True
    check('infeasible_budget_rejected',infeasible)
    try:allocate(a,20,floor=.2,reserve=(np.array([0]),4));intersect=False
    except ValueError:intersect=True
    check('unsupported_intersecting_constraints_rejected',intersect)
    al=pd.read_csv(OUT/'optimization_allocations.csv');group=al.groupby(['scenario','budget']).allocated.sum()
    check('all_707_budget_runs',len(group)==707);check('every_budget_exact',all(value==budget for (sc,budget),value in group.items()));check('every_cell_minimum_5',al.allocated.min()>=5)
    cv=pd.read_csv(OUT/'optimization_curves.csv');check('coverage_nondecreasing_each_scenario',all((g.coverage.diff().dropna()>=-1e-12).all() for _,g in cv.groupby('scenario')))
    par=pd.read_csv(OUT/'optimization_pareto.csv');check('pareto_health_nondecreasing',(par.health_fraction.diff().dropna()>=-1e-10).all());check('pareto_coverage_nonincreasing',(par.coverage.diff().dropna()<=1e-10).all())
    fp=pd.read_csv(OUT/'spatial_footprint.csv');check('footprint_sums_one',np.isclose(fp.footprint_share.sum(),1))
    fed=json.loads((OUT/'federated_protocol.json').read_text('utf-8'));check('federation_exact_count_and_parameters',fed['exact_count_max_gap']==0 and fed['exact_CPD_max_gap']==0);check('federation_identical_allocation',fed['federated_allocation_identical'])
    check('31_clients_30_resolved',fed['clients']==31 and fed['resolved_province_clients']==30);check('DP_140_repetitions',len(pd.read_csv(OUT/'federated_dp_runs.csv'))==140)
    check('DP_report_sensitivity_30',fed['count_L1_sensitivity']==30)
    # Independent add/remove report perturbation checks the stated contribution cap.
    bn=bn_data();rid=bn.groupby('record_id').size().idxmax();deleted=m.counts(bn[~bn.record_id.eq(rid)]);delta=sum(abs(m.count[n]-deleted[n]).sum() for n in NODES)
    check('DP_sensitivity_attained_and_bounded',delta==30,delta)
    for name in [*[f'Figure_{j}' for j in range(1,9)],*[f'Supplementary_Figure_S{j}' for j in range(1,5)],'Graphical_Abstract']:
        for ext in ['png','pdf','svg']:check(name+'_'+ext,(FIG/(name+'.'+ext)).is_file())
        with Image.open(FIG/(name+'.png')) as im:check(name+'_native_resolution',min(im.size)>3000 and min(im.info.get('dpi',(0,0)))>599)
    chronology=pd.read_csv(OUT/'lead_chronology_audit.csv');dates=pd.to_datetime(chronology.notification_date)
    check('lead_all_22_reports_retained',len(chronology)==22 and chronology.record_id.nunique()==22)
    check('lead_20_earlier_two_recent',(dates<pd.Timestamp('2021-09-21')).sum()==20 and (dates>=pd.Timestamp('2021-09-21')).sum()==2)
    periods=pd.read_csv(OUT/'period_category_audit.csv');check('period_audit_all_categories_retained',len(periods)==16)
    for cat in CATS:
        check('period_total_'+cat,periods.loc[periods.category.eq(cat),'reports_with_category'].sum()==d.loc[d.category_en.eq(cat),'record_id'].nunique())
    save(pd.DataFrame(checks),'validation_checks_full');passed=all(x['passed'] for x in checks)
    jsave({'passed':passed,'n_checks':len(checks),'checks':checks},'validation_full')
    if not passed:raise SystemExit('Validation failed')
    print('Passed',len(checks),'numerical and artifact checks.')

if __name__=='__main__':main()
