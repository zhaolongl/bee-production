"""Direct-consumption, NONCOMPLIANT-RECORD-CONDITIONAL exposure experiments.

No infusion/leaching parameter exists. Intake amounts are explicit analyst
scenarios, not observed Chinese consumption distributions. No borrowed ADI
is assigned to a drug without a verified applicable reference value.
"""
import itertools
import numpy as np
import pandas as pd
from bee_common import *

WHO='https://apps.who.int/food-additives-contaminants-jecfa-database/Home/Chemical/'
# ADI mg/kg bw/day; numeric values are upper bounds of ADI ranges.
PARAM = {
 'Sorbates':(25.,'JECFA 1973',WHO+'2443','Sorbic acid equivalents'),
 'Benzoates':(20.,'JECFA 2021',WHO+'1227','Benzoic acid equivalents'),
 'Acesulfame K':(15.,'JECFA 1990',WHO+'926','Acesulfame potassium'),
 'Cyclamate':(11.,'JECFA 2009',WHO+'995','Cyclamic acid equivalents; ambiguous legacy aliases use an upper-bound mass assumption'),
 'Sodium saccharin':(9.,'EFSA 2024','https://doi.org/10.2903/j.efsa.2024.9044','Free imide; reported sodium salt mass treated as free imide is an upper-bound screening assumption'),
 'Sunset yellow':(4.,'JECFA 2011',WHO+'476','Colouring component; aluminium not assessed'),
 'Tartrazine':(10.,'JECFA 2016',WHO+'477','Colouring component; aluminium not assessed'),
 'Amitraz':(.01,'JMPR 1998','https://www.inchem.org/documents/jmpr/jmpmono/v098pr02.htm','Reported amitraz; full residue definition not demonstrable'),
 'Tau-fluvalinate':(.005,'EFSA 2021','https://doi.org/10.2903/j.efsa.2021.6646','Reported tau-fluvalinate'),
}
INTAKE={'Honey':(10.,5.,30.),'Royal jelly':(3.,1.,5.),'Bee pollen':(5.,2.,10.),'Other bee products':(10.,5.,30.)}
BW=60.; ITERATIONS=20000

def edi(conc_ugkg, intake_g, bw=BW): return conc_ugkg*intake_g/(1000*bw)

def tox_score(adi,category):
    # Lower ADI -> higher score, interpolated within each band.
    t=np.interp(adi,[0,.0001,.005,.1,5],[5,4,3,2,1])
    lo,hi=(25,38) if category in ['Pesticides','Veterinary drugs'] else (10,25)
    return lo+(t-1)/4*(hi-lo)

def exposure():
    r,i,d=load_data();q=d[d.quantitative_primary.eq(True)].copy()
    rng=np.random.default_rng(SEED); rows=[]; grid=[]; hist=[]; lead=[]; params=[]; critical=[]
    for item in sorted(d.item_en.unique()):
        p=PARAM.get(item)
        reason='Applicable ADI available' if p else 'No verified applicable ADI; retained in occurrence, network and allocation analyses'
        if item=='Chloramphenicol':reason='JECFA: no ADI allocated; EDI only'
        if item=='Lead':reason='No ADI; adult kidney/BP BMDL-based MOE evaluated separately'
        if item=='Tetracyclines':reason='Group identity unresolved: JECFA 0.03 applies only to specified tetracyclines; no automatic class-wide assignment'
        params.append({'item':item,'adi_mgkg_bw_day':p[0] if p else np.nan,'authority':p[1] if p else '',
            'url':p[2] if p else (WHO+'551' if item=='Chloramphenicol' else ''),'basis':p[3] if p else '',
            'status':reason,'accessed':'2026-09-20'})
    save(pd.DataFrame(params),'toxicology_parameters')
    save(pd.DataFrame([{'product':k,'baseline_g_day':v[0],'uniform_min_g_day':v[1],'uniform_max_g_day':v[2],
                       'body_weight_kg':BW,'basis':'Investigator-defined scenario; not an observed intake distribution'} for k,v in INTAKE.items()]),'intake_scenarios')
    for (item,product),g in q.groupby(['item_en','product_en']):
        c=g.concentration_ugkg.to_numpy(float);di,lo,hi=INTAKE[product];p=PARAM.get(item);adi=p[0] if p else np.nan
        sim=edi(rng.choice(c,ITERATIONS),rng.uniform(lo,hi,ITERATIONS))
        hq=sim/(1000*adi)
        row={'item':item,'product':product,'n':len(c),'concentration_median_ugkg':np.median(c),'concentration_max_ugkg':max(c),
             'edi_baseline_ugkg_bw_day':edi(np.median(c),di),'edi_max_baseline':edi(max(c),di),
             'adi_mgkg_bw_day':adi,'hq_baseline':edi(np.median(c),di)/(1000*adi),
             'hq_max_baseline':edi(max(c),di)/(1000*adi),
             'edi_sim_median':np.median(sim),'edi_sim_p95':np.quantile(sim,.95),
             'hq_sim_median':np.median(hq),'hq_sim_p95':np.quantile(hq,.95),
             'p_hq_gt1':float(np.mean(hq>1)) if p else np.nan,
             'small_n_flag':len(c)<5,'mass_basis_upper_bound':item in ['Sodium saccharin','Cyclamate'],
             'scenario_distribution_not_CI':True}
        rows.append(row)
        # Four BW x five intake multipliers x two concentration summaries.
        for bw,mul,cstat in itertools.product([45.,60.,75.,90.],[.5,1,2,3,5],['median','maximum']):
            cc=np.median(c) if cstat=='median' else max(c);ee=edi(cc,di*mul,bw)
            grid.append({'item':item,'product':product,'body_weight_kg':bw,'intake_multiplier':mul,'intake_g_day':di*mul,'concentration_basis':cstat,'edi':ee,'hq':ee/(1000*adi)})
        if p:
            for bw,stat in itertools.product([45.,60.,75.,90.],['mean','median','maximum']):
                cc={'mean':np.mean(c),'median':np.median(c),'maximum':max(c)}[stat]
                critical.append({'item':item,'product':product,'n':len(c),'body_weight_kg':bw,
                    'concentration_basis':stat,'concentration_ugkg':cc,'adi_mgkg_bw_day':adi,
                    'critical_intake_g_day':1e6*bw*adi/cc,'baseline_intake_g_day':di,
                    'multiple_of_baseline_intake':1e6*bw*adi/(cc*di),
                    'interpretation':'Algebraic intake at HQ=1 for the reported concentration summary; not a recommended intake.'})
            bins=np.geomspace(max(hq.min()*.99,1e-10),max(hq.max()*1.01,1e-9),61)
            counts,edges=np.histogram(hq,bins)
            hist.extend({'item':item,'product':product,'hq_bin_left':a,'hq_bin_right':b,'count':int(n)} for a,b,n in zip(edges[:-1],edges[1:],counts))
        if item=='Lead':
            for endpoint,bmdl in [('Adult kidney',.63),('Adult systolic blood pressure',1.50)]:
                moe=bmdl/sim
                lead.append({'product':product,'endpoint':endpoint,'n':len(c),'bmdl_ugkg_bw_day':bmdl,'baseline_moe':bmdl/row['edi_baseline_ugkg_bw_day'],
                  'moe_sim_p05':np.quantile(moe,.05),'moe_sim_median':np.median(moe),'p_moe_lt1':np.mean(moe<1),'reference':'https://doi.org/10.2903/j.efsa.2010.1570'})
    e=save(pd.DataFrame(rows),'exposure');save(pd.DataFrame(grid),'exposure_grid');save(pd.DataFrame(hist),'exposure_histogram');save(pd.DataFrame(lead),'lead_moe')
    save(pd.DataFrame(critical),'critical_intake')
    # Same-report screening sums, never independent marginal medians passed off as co-exposure.
    q['adi']=q.item_en.map({k:v[0] for k,v in PARAM.items()})
    q['intake']=q.product_en.map({k:v[0] for k,v in INTAKE.items()})
    q['hq']=edi(q.concentration_ugkg,q.intake)/(1000*q.adi)
    subset=q[q.adi.notna()]
    sums=subset.groupby(['record_id','product_en']).agg(measured_components=('item_en','nunique'),screening_sum_HQ=('hq','sum')).reset_index()
    sums['interpretation']='Same-report measured-subset screening sum; dose-additivity assumption; no validated common-mechanism mixture HI'
    save(sums,'mixture_screening')
    # Sum of marginal medians across separate reports: a hypothetical simultaneous exposure.
    co=e[e.adi_mgkg_bw_day.notna()].groupby('product').agg(sum_marginal_median_HQ=('hq_baseline','sum'),components=('item','nunique')).reset_index()
    co['interpretation']='Hypothetical simultaneous exposure to separate noncompliant median concentrations; not observed co-occurrence'
    save(co,'hypothetical_coexposure')
    # Published benchmark alternatives; test source sensitivity rather than treating guidance as timeless.
    alternatives={'Sorbates':(11.,'https://doi.org/10.2903/j.efsa.2019.5625'),
                  'Benzoates':(5.,'https://doi.org/10.2903/j.efsa.2016.4433'),
                  'Sodium saccharin':(5.,WHO+'3164')}
    alt=[]
    for item,(adi,url) in alternatives.items():
        for _,a in e[e.item==item].iterrows():alt.append({'item':item,'product':a['product'],'primary_adi':a.adi_mgkg_bw_day,'alternative_adi':adi,'alternative_hq':a.edi_baseline_ugkg_bw_day/(1000*adi),'url':url})
    save(pd.DataFrame(alt),'benchmark_sensitivity')
    jsave({'draws_per_item_product':ITERATIONS,'seed':SEED,'exposure_scope':'conditional on quantified noncompliant records',
        'no_transfer_factor':True,'intake_basis':'investigator-defined scenarios','ADI_hazards':len(PARAM),'quantified_rows':len(q)},'exposure_protocol')
    return e

def ranking(e):
    r,i,d=load_data();rows=[]
    hq=e.groupby('item').hq_baseline.max().to_dict() # worst assessed product median; transparent screening choice
    # Source-reported limit language, not a retrospective claim of intentional addition.
    for item,g in d.groupby('item_en'):
        cat=g.category_en.iloc[0];pc=g.record_id.nunique()/NT
        nature=float(g.limit_raw.fillna('').astype(str).str.contains('不得使用|不得添加|不得检出|不能检出|禁止').mean())
        ns=20*nature;ps=np.interp(pc*100,[0,.005,.020,.040,.070,.4],[0,10,20,25,30,40])
        adi=PARAM.get(item,(np.nan,))[0];ts=tox_score(adi,cat) if np.isfinite(adi) else np.nan
        hs=np.interp(hq.get(item,np.nan),[0,.01,.1,.5,1,2],[0,10,20,28,34,40])
        reg=(.5*ts+.3*ps+.2*ns)/36*100
        hea=(.3*ts+.3*hs+.25*ps+.15*ns)/37*100
        rows.append({'item':item,'category':cat,'records':g.record_id.nunique(),'pc':pc,'adi':adi,'max_product_median_hq':hq.get(item,np.nan),
          'toxicity_score':ts,'occurrence_score':ps,'restriction_score':ns,'hq_score':hs,'regulatory_score':reg,'health_score':hea,
          'regulatory_lower_if_toxicity_unknown':(.3*ps+.2*ns)/36*100 if np.isnan(ts) else reg,
          'regulatory_upper_if_toxicity_unknown':(.5*40+.3*ps+.2*ns)/36*100 if np.isnan(ts) else reg,
          'restricted_limit_fraction':nature,'grading_status':'Graded with applicable ADI' if np.isfinite(adi) else 'Unrated; absence of score is not low risk'})
    rank=pd.DataFrame(rows)
    for col in ['regulatory','health']:
        rank[col+'_rank']=rank[col+'_score'].rank(ascending=False,method='min')
        rank[col+'_grade']=pd.cut(rank[col+'_score'],[-1,20,35,50,65,101],right=False,labels=['Low','Moderately low','Moderate','Moderately high','High']).astype(str).replace('nan','Unrated')
    rank=save(rank.sort_values('regulatory_score',ascending=False,na_position='last'),'risk_ranking')
    ok=rank.dropna(subset=['health_score']).copy();weights=[];ranks=[]
    for wt,we,wp in itertools.product(range(2,15),repeat=3):
        wb=20-wt-we-wp
        if not 1<=wb<=7:continue
        w=np.array([wt,we,wp,wb])/20
        score=ok[['toxicity_score','hq_score','occurrence_score','restriction_score']].to_numpy()@w
        rk=pd.Series(score,index=ok.item).rank(ascending=False,method='min')
        weights.append(dict(zip(['toxicity','exposure','occurrence','restriction'],w)))
        ranks.append(rk)
    rr=pd.DataFrame(ranks)
    sensitivity=pd.DataFrame({'item':rr.columns,'rank_min':rr.min(),'rank_median':rr.median(),'rank_max':rr.max(),'top5_fraction':(rr<=5).mean(),
       'baseline_health_rank':ok.set_index('item').health_rank.reindex(rr.columns)}).reset_index(drop=True)
    save(sensitivity,'risk_weight_sensitivity');save(pd.DataFrame(weights),'risk_weight_grid')
    prod=[]
    for product,g in d.groupby('product_en'):
        for basis in ['regulatory','health']:
            scores=g.item_en.map(rank.set_index('item')[basis+'_score'])
            mean=scores.mean();share=r.product_en.eq(product).mean()
            prod.append({'product':product,'basis':basis,'eligible_events':scores.notna().sum(),'all_events':len(g),'coverage_fraction':scores.notna().mean(),
                         'mean_hazard_score':mean,'failure_report_share':share,'product_composite':.7*mean+.3*share*100,
                         'interpretation':'0.7 x mean item score over the items with an applicable ADI + 0.3 x share of non-compliant reports (x 100)'})
    save(pd.DataFrame(prod),'product_risk_ranking')
    jsave({'weight_combinations':len(weights),'n_graded':len(ok),'n_all_items':len(rank),
           'weights_regulatory':[.5,.3,.2],'weights_health':[.3,.3,.25,.15],
           'not_graded_reason':'No applicable verified ADI; full-scope records retained elsewhere'},'ranking_protocol')

if __name__=='__main__':ranking(exposure())
