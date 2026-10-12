"""Figures 1-8, Supplementary Figures S1-S4 and the graphical abstract.

All charts are generated from the exported results. Each figure is written as
600-dpi PNG and editable PDF/SVG, with a thumbnail in figures/previews; set
BEE_TIFF=1 to add 600-dpi LZW-compressed TIFF.
"""
import json
import os
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.ticker import PercentFormatter, MaxNLocator
from bee_common import *

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','axes.linewidth':.8})
BLUE,ORANGE,GREEN,RED,PURPLE='#3b76af','#e1812c','#3a923a','#c95a5b','#9467bd'
CC=dict(zip(CATS,COLORS));MANIFEST=[]
RC=dict(zip(REGIONS,[BLUE,ORANGE,GREEN,PURPLE,'#8c564b','#777777','#17a2a4',RED]));RC['Unknown']=RED
def read(n):return pd.read_csv(OUT/(n+'.csv'))
def short(x):return SHORT.get(x,x)
def label(ax,letter,title=None):
    pos=ax.get_position();xx=max(-.12,(.012-pos.x0)/pos.width)
    ax.text(xx,1.07,f'({letter})',transform=ax.transAxes,fontweight='bold',fontsize=14,va='top')
    if title:ax.set_title(title,loc='left',pad=12)
def grid(ax,axis='y'):ax.grid(axis=axis,ls=':',alpha=.25);ax.set_axisbelow(True)
def pct(ax,axis='y'):
    (ax.yaxis if axis=='y' else ax.xaxis).set_major_formatter(PercentFormatter(1))
def savefig(fig,name,sources,alt):
    FIG.mkdir(exist_ok=True);(FIG/'previews').mkdir(exist_ok=True)
    fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.svg'))
    fig.savefig(FIG/(name+'.png'),dpi=600)
    if os.environ.get('BEE_TIFF'):fig.savefig(FIG/(name+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(FIG/'previews'/(name+'.png'),dpi=115)
    MANIFEST.append({'figure':name,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),'dpi':600,
       'sources':sources,'alt_text':alt,'seed':SEED})
    plt.close(fig);print('Exported',name,flush=True)

def fig1():
    fig,ax=plt.subplots(figsize=(13.2,7.1));fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color,size=11):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.007,rounding_size=.012',fc=color,ec='#626262',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.5)
    heads=['Occurrence and heterogeneity','Bayesian spatial inference','Direct-consumption exposure','Monitoring and collaboration']
    cols=['#fce9b8','#cfe3f3','#d9edd3','#e4daf0'];xx=[.018,.268,.518,.768]
    datas=['National / provincial / city reports\n87,313 monitored batches\n1,798 noncompliant reports',
           'Labelled production location\nSampling location and stage\nFour bee-product categories',
           'Reported chemical concentrations\nVerified toxicological benchmarks\nExplicit intake and body-weight scenarios',
           'Item-stratum posterior probabilities\nInteger budgets and coverage floors\nProvince-partitioned count tables']
    methods=['All 46 standardized items\nMatched product denominators\nYear, region, product and stage',
             'Six-node TAN, BDeu estimation\nGrouped holdout and five-fold CV\nPosterior footprint and scale ablation',
             'EDI = C × DI / (1000 × BW)\nEDI, HQ and cumulative screening\n20,000 draws; parameter scenarios',
             'Seven scenarios; 101 budget points\n473 ranking-weight combinations\nFederation; central-DP simulation']
    outcomes=['Reported occurrence and PC(k)', 'P(stratum | item); spatial footprint', 'EDI; HQ; benchmark coverage', 'Integer allocation and utility trade-offs']
    for j,x in enumerate(xx):
        box(x,.80,.214,.155,datas[j],'#ececec',9.2)
        box(x,.69,.214,.065,f'{j+1}. '+heads[j],cols[j],9.5)
        box(x,.36,.214,.27,methods[j],cols[j],10)
        box(x,.22,.214,.085,outcomes[j],cols[j],9)
        for y1,y2 in [(.80,.762),(.69,.638),(.36,.314),(.22,.15)]:ax.annotate('',(x+.107,y2),(x+.107,y1),arrowprops={'arrowstyle':'->','color':'#555555','lw':1.1})
        if j<3:ax.annotate('',(xx[j+1]-.01,.49),(x+.224,.49),arrowprops={'arrowstyle':'->','color':'#555555','lw':1.2})
    box(.025,.045,.944,.10,'Objective: auditable allocation of bee-product inspection resources\nHistorical footprint coverage, exposure-screening priorities and collaboration costs','#f6d6d6',12)
    savefig(fig,'Figure_1',['bn_protocol.json','exposure_protocol.json','optimization_protocol.json'],'Four-stage framework: occurrence and heterogeneity, Bayesian spatial inference, direct-consumption exposure, and monitoring and collaboration.')

def stacked(ax,tab,orient='bar',legend=False):
    base=np.zeros(len(tab))
    for cat in CATS:
        vals=tab[cat].to_numpy() if cat in tab else base*0
        if orient=='bar':ax.bar(np.arange(len(tab)),vals,bottom=base,color=CC[cat],label=cat,width=.72)
        else:ax.barh(np.arange(len(tab)),vals,left=base,color=CC[cat],label=cat,height=.7)
        base+=vals

def fig2():
    r,i,d=load_data();fig,axs=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(left=.09,right=.985,top=.94,bottom=.19,hspace=.59,wspace=.43)
    a=read('annual');ax=axs[0,0];ax.bar(a.year,a.failed,color=BLUE);ax.set(ylabel='Noncompliant reports',xlabel='Notification year',xticks=a.year);ax.tick_params(axis='x',rotation=45)
    for x,n,v in zip(a.year,a.failed,a.rate):ax.text(x,n+7,f'{v:.1%}',ha='center',fontsize=8)
    ax.set_ylim(0,a.failed.max()*1.25);label(ax,'a','Annual reports and observed fractions');grid(ax)
    ax=axs[0,1];tab=pd.crosstab(d.year,d.category_en).reindex(columns=CATS,fill_value=0);tab=tab.div(tab.sum(axis=1),axis=0)
    ax.stackplot(tab.index,tab.T,colors=[CC[x] for x in CATS],labels=CATS);ax.set(xlabel='Notification year',ylabel='Share of item events',ylim=(0,1));pct(ax);label(ax,'b','Temporal hazard composition')
    ax=axs[0,2];q=d[d.quantitative_primary.eq(True)];top=q.item_en.value_counts().head(8).index
    vals=[q.loc[q.item_en.eq(it),'concentration_ugkg'].to_numpy() for it in top]
    bp=ax.boxplot(vals,vert=False,patch_artist=True,widths=.58,flierprops={'markersize':2,'alpha':.35})
    for b in bp['boxes']:b.set(facecolor='#cfe3f3')
    ax.set(yticklabels=[f'{short(it)} (n={len(v)})' for it,v in zip(top,vals)],xscale='log',xlabel='Concentration (µg/kg; log scale)');ax.invert_yaxis();label(ax,'c','Reported quantitative chemical results');grid(ax,'x')
    ax=axs[1,0];p=pd.read_csv(DATA/'province.csv').groupby('region')[['total','failed']].sum().reindex(REGIONS).dropna();v=p.failed/p.total
    ax.barh(np.arange(len(v)),v,color=[RED if x=='Other / unknown' else BLUE for x in p.index]);ax.set(yticks=np.arange(len(v)),yticklabels=p.index,xlabel='Noncompliant / monitored batches');ax.invert_yaxis();pct(ax,'x');ax.set_xlim(0,v.max()*1.35)
    for j,z in enumerate(v):ax.text(z+.0004,j,f'{z:.2%}',va='center',fontsize=8)
    label(ax,'d','Observed fractions by production region');grid(ax,'x')
    ax=axs[1,1];t=pd.crosstab(d.product_en,d.category_en).reindex(PRODUCTS).fillna(0);t=t.div(t.sum(axis=1),axis=0);stacked(ax,t)
    ax.set(xticks=range(4),xticklabels=['Honey','Royal jelly','Bee pollen','Other\nproducts'],ylabel='Share of item events',ylim=(0,1));pct(ax);label(ax,'e','Product-specific hazard composition')
    ax=axs[1,2];b=bn_data();t1=pd.crosstab(b.Stage,b.HazardCat);t2=pd.crosstab(b.ProdR,b.HazardCat).reindex(REGIONS);t=pd.concat([t1,t2]).fillna(0);t=t.div(t.sum(axis=1),axis=0);stacked(ax,t)
    ax.set(xticks=range(len(t)),xticklabels=[x.replace('Other / unknown','Other / unk.') for x in t.index],ylabel='Share of item events',ylim=(0,1));ax.tick_params(axis='x',rotation=65,labelsize=7);ax.axvline(len(t1)-.5,c='#555',ls='--',lw=.8);pct(ax);label(ax,'f','Stages (left) and production regions (right)')
    handles,labels=axs[0,1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.065),frameon=False)
    fig.text(.5,.025,'2015–2025 are complete years; 2026 ends on 20 September. Composition panels use 1,920 unique report–item events.',ha='center',fontsize=9)
    savefig(fig,'Figure_2',['annual.csv','records.csv','items.csv','province.csv'],'Six occurrence panels covering all bee products and all reported hazard categories.')

def fig3():
    from bee_bn import load_model
    m=load_model();fig=plt.figure(figsize=(15.8,9));gs=fig.add_gridspec(2,3,height_ratios=[.78,1.35],left=.13,right=.98,bottom=.24,top=.95,hspace=.45,wspace=.60)
    ax=fig.add_subplot(gs[0,:]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');label(ax,'a','Tree-augmented Bayesian network')
    pos={'HazardCat':(.12,.73),'Item':(.38,.73),'Product':(.67,.77),'ProdR':(.42,.2),'SampR':(.69,.2),'Stage':(.92,.44)}
    patches={}
    for n,(x,y) in pos.items():
        patch=FancyBboxPatch((x-.047,y-.105),.094,.21,boxstyle='round,pad=.009',fc='#cfe3f3' if n!='Item' else '#fce9b8',ec='#5e6b76',zorder=3);ax.add_patch(patch);patches[n]=patch
        ax.text(x,y,{'HazardCat':'Hazard\ncategory','Item':'Hazard\nidentity','Product':'Product\ntype','ProdR':'Production\nregion','SampR':'Sampling\nregion','Stage':'Supply-chain\nstage'}[n],ha='center',va='center',fontsize=10,zorder=4)
    for child,parents in m.parents.items():
        for parent in parents:
            start,end=pos[parent],pos[child]
            patch=FancyArrowPatch(start,end,patchA=patches[parent],patchB=patches[child],arrowstyle='-|>',mutation_scale=13,shrinkA=2,shrinkB=2,color='#9a9a9a',lw=1.2,connectionstyle='arc3,rad=.07' if parent=='HazardCat' else 'arc3,rad=0',zorder=2)
            ax.add_patch(patch)
    ax.text(.98,.02,'Item marginalized during classification; arrows encode conditional structure.',ha='right',fontsize=9,style='italic')
    ax=fig.add_subplot(gs[1,0]);p=read('bn_item_by_sampling_region');top=read('item_summary').sort_values('records_with_item',ascending=False).item_en.head(10)
    tab=p.pivot(index='item',columns='sampling_region',values='probability').reindex(top)
    im=ax.imshow(tab,aspect='auto',cmap='Blues',vmin=0,vmax=tab.max().max());ax.set(yticks=range(len(tab)),yticklabels=[short(x) for x in tab.index],xticks=range(len(tab.columns)),xticklabels=tab.columns);ax.tick_params(axis='x',rotation=60);label(ax,'b','P(item | sampling region)');fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='Conditional probability')
    ax=fig.add_subplot(gs[1,1]);p=m.query(['ProdR','SampR'],{'HazardCat':'Veterinary drugs'});p=p/p.sum(axis=0,keepdims=True);base=np.zeros(p.shape[1])
    for j,region in enumerate(m.states['ProdR']):ax.bar(range(p.shape[1]),p[j],bottom=base,color=RC[region],label=region);base+=p[j]
    ax.set(xticks=range(p.shape[1]),xticklabels=m.states['SampR'],ylabel='Posterior production-region share',ylim=(0,1));ax.tick_params(axis='x',rotation=60);pct(ax);label(ax,'c','Veterinary-drug production locations');handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,.035),ncol=4,frameon=False,fontsize=9)
    ax=fig.add_subplot(gs[1,2]);v=read('bn_validation');names=['Accuracy','Balanced accuracy','Macro-F1'];cols=['accuracy','balanced_accuracy','macro_f1'];xx=np.arange(3)
    for j,(scale,col) in enumerate([('Region',BLUE),('Province',ORANGE)]):
        cv=v[v.scale.eq(scale)&v.split.str.startswith('CV')];mean=cv[cols].mean();sd=cv[cols].std()
        ax.bar(xx+(j-.5)*.34,mean,width=.32,yerr=sd,color=col,label=scale,capsize=3)
    ax.set(xticks=xx,xticklabels=['Accuracy','Balanced\naccuracy','Macro-F1'],ylabel='Five-fold grouped validation',ylim=(0,.7));ax.legend(frameon=False);label(ax,'d','Spatial-resolution ablation');grid(ax)
    savefig(fig,'Figure_3',['bn_protocol.json','bn_item_by_sampling_region.csv','bn_validation.csv'],'Network above three inference and scale-comparison panels. Error bars are between-fold standard deviations.')

def sankey(ax,tab):
    # No links are filtered or renormalized. Both ends retain the same denominator.
    left=list(tab.index);right=list(tab.columns);mat=tab.to_numpy();gap=.024;usable=1-gap*(max(len(left),len(right))-1)
    totals_l=mat.sum(1);totals_r=mat.sum(0)
    y0={};y1={};cursor=0
    for s,v in zip(left,totals_l):y0[s]=cursor;cursor+=v*usable+gap
    cursor=0
    for s,v in zip(right,totals_r):y1[s]=cursor;cursor+=v*usable+gap
    ol={s:0. for s in left};orr={s:0. for s in right}
    for i,s in enumerate(left):
        color=RC[s]
        for j,t in enumerate(right):
            h=mat[i,j]*usable;a=y0[s]+ol[s];b=y1[t]+orr[t];ol[s]+=h;orr[t]+=h
            vertices=[(.12,a),(.42,a),(.58,b),(.88,b),(.88,b+h),(.58,b+h),(.42,a+h),(.12,a+h),(.12,a)]
            ax.add_patch(PathPatch(MPath(vertices,[1,4,4,4,2,4,4,4,79]),fc=color,ec='none',alpha=.28))
    for side,labels,totals,positions,x in [('left',left,totals_l,y0,.105),('right',right,totals_r,y1,.88)]:
        for j,(s,v) in enumerate(zip(labels,totals)):
            c=RC[s]
            ax.add_patch(Rectangle((x,positions[s]),.015,v*usable,fc=c,ec='none'))
            ax.text(x-.015 if side=='left' else x+.03,positions[s]+v*usable/2,f'{s}  {v:.1%}',ha='right' if side=='left' else 'left',va='center',fontsize=9)
    ax.set(xlim=(-.09,1.13),ylim=(-.015,1.04));ax.invert_yaxis();ax.axis('off')

def fig4():
    p=read('spatial_category_footprints');f=read('spatial_footprint');tab=f.pivot(index='production_region',columns='sampling_region',values='footprint_share').reindex(REGIONS).fillna(0)
    fig=plt.figure(figsize=(15.8,8.6));gs=fig.add_gridspec(2,3,height_ratios=[1,1.25],left=.09,right=.93,top=.94,bottom=.055,hspace=.43,wspace=.56)
    ax=fig.add_subplot(gs[0,0]);s=p.groupby('category').footprint_share.sum().sort_values();ax.barh(s.index,s,color=[CC[x] for x in s.index]);pct(ax,'x');ax.set_xlabel('Share of modelled event footprint');label(ax,'a','Hazard-category contributions');grid(ax,'x')
    ax=fig.add_subplot(gs[0,1]);s=f.groupby('sampling_region').footprint_share.sum().sort_values();unknown=f[f.production_region.eq('Other / unknown')].set_index('sampling_region').footprint_share.reindex(s.index);ax.barh(s.index,s-unknown,color=BLUE,label='Domestic production location');ax.barh(s.index,unknown,left=s-unknown,color=RED,label='Other / unknown production');pct(ax,'x');label(ax,'b','Sampling regions and source availability');grid(ax,'x');ax.legend(frameon=False,fontsize=7,loc='lower right')
    ax=fig.add_subplot(gs[0,2]);im=ax.imshow(tab,cmap='Blues',aspect='auto');ax.set(yticks=range(len(tab)),yticklabels=tab.index,xticks=range(len(tab.columns)),xticklabels=tab.columns);ax.tick_params(axis='x',rotation=60);label(ax,'c','Production–sampling footprint');fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label='Share of total footprint')
    ax=fig.add_subplot(gs[1,:]);sankey(ax,tab);label(ax,'d','Posterior-weighted production–sampling links (all links retained)')
    savefig(fig,'Figure_4',['spatial_category_footprints.csv','spatial_footprint.csv'],'Category and regional shares, posterior matrix and a full-width alluvial diagram; these are report links, not trade or contaminant flows.')

def fig5():
    e=read('exposure');valid=e[e.adi_mgkg_bw_day.notna()];e1=valid.loc[valid.groupby('item').hq_baseline.idxmax()].sort_values('hq_baseline',ascending=False)
    fig,aa=plt.subplots(2,2,figsize=(15.3,10));fig.subplots_adjust(left=.14,right=.975,top=.945,bottom=.11,wspace=.35,hspace=.48)
    ax=aa[0,0];y=np.arange(len(e1));ax.hlines(y,e1.hq_baseline,e1.hq_max_baseline,color='#bbbbbb',lw=3)
    ax.scatter(e1.hq_baseline,y,c=BLUE,edgecolor='#555555',label='Median concentration',zorder=3)
    ax.scatter(e1.hq_max_baseline,y,c=ORANGE,edgecolor='#555555',label='Maximum concentration',zorder=3)
    ax.axvline(1,color=RED,ls='--');ax.set(yticks=y,yticklabels=[f'{short(x)} (n={n})' for x,n in zip(e1.item,e1.n)],xscale='log',xlabel='Conditional HQ (log scale)',xlim=(1e-5,2));ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8,frameon=False);label(ax,'a','Baseline direct-consumption screening');grid(ax,'x')
    ax=aa[0,1];top=e1.head(3);xx=np.arange(4);scenarios=[(.5,60),(1,60),(2,60),(3,45)]
    for j,(_,row) in enumerate(top.iterrows()):
        yy=[row.hq_baseline*mul*60/bw for mul,bw in scenarios];ax.bar(xx+(j-1)*.24,yy,.23,color=[BLUE,ORANGE,GREEN][j],label=short(row['item']))
    ax.set(xticks=xx,xticklabels=['0.5 × DI\n60 kg','1 × DI\n60 kg','2 × DI\n60 kg','3 × DI\n45 kg'],ylabel='HQ at median concentration');ax.legend(frameon=False);label(ax,'b','Intake and body-weight scenarios');grid(ax)
    cr=read('critical_intake');cr=cr[cr.body_weight_kg.eq(60)&cr.concentration_basis.eq('mean')].sort_values('critical_intake_g_day').iloc[0]
    ax.set_ylim(0,ax.get_ylim()[1]*1.3)
    ax.text(.025,.965,f'Minimum DI at HQ = 1: {cr.critical_intake_g_day:,.0f} g/day\n60 kg; mean C; {cr["item"].lower()} in {cr["product"].lower()}',transform=ax.transAxes,va='top',fontsize=8.3,bbox={'fc':'#fff5dc','ec':'#bfa671','pad':4})
    ax=aa[1,0];s=valid.loc[valid.groupby('item').hq_sim_p95.idxmax()].sort_values('hq_sim_p95',ascending=False)
    ax.barh(np.arange(len(s)),s.hq_sim_p95,color=PURPLE);ax.set(yticks=range(len(s)),yticklabels=[short(x) for x in s.item],xscale='log',xlim=(1e-5,2),xlabel='95th-percentile scenario HQ (log scale)');ax.invert_yaxis();ax.axvline(1,color=RED,ls='--')
    for j,v in enumerate(s.p_hq_gt1):ax.text(.95,j,f'P(HQ > 1) = {v:.2%}',va='center',ha='right',transform=ax.get_yaxis_transform(),fontsize=8)
    label(ax,'c','20,000 conditional exposure draws');grid(ax,'x')
    ax=aa[1,1];co=read('hypothetical_coexposure');ax.barh(range(len(co)),co.sum_marginal_median_HQ,color=GREEN,height=.5)
    ax.set(yticks=range(len(co)),yticklabels=co['product'],xlabel='Hypothetical sum of marginal median HQs');ax.set_xlim(0,co.sum_marginal_median_HQ.max()*1.4)
    for j,(_,row) in enumerate(co.iterrows()):ax.text(row.sum_marginal_median_HQ+.00015,j,f'{row.sum_marginal_median_HQ:.4f}\n{row.components} components',va='center',fontsize=9)
    ax.set_ylim(-.8,1.4)
    ax.text(.02,.025,'Assumed simultaneous exposure to separate reports.\nNo validated common-mechanism HI is inferred.\nAdditional exposure estimates: Fig. S1.',transform=ax.transAxes,fontsize=9,bbox={'fc':'#f7f7f7','ec':'none','pad':5})
    label(ax,'d','Cumulative-screening experiment');grid(ax,'x')
    fig.text(.5,.03,'All exposure results are conditional scenarios from noncompliant records; intake distributions are analyst-defined, not population estimates.',ha='center',fontsize=9)
    savefig(fig,'Figure_5',['exposure.csv','hypothetical_coexposure.csv','intake_scenarios.csv','critical_intake.csv'],'Direct-consumption exposure, intake and body-weight sensitivity, conditional scenario quantiles and hypothetical co-exposure among chemicals with applicable ADIs.')

def fig6():
    r=read('risk_ranking').dropna(subset=['health_score']);sen=read('risk_weight_sensitivity');pr=read('product_risk_ranking')
    fig,aa=plt.subplots(2,2,figsize=(15.2,10));fig.subplots_adjust(left=.14,right=.96,top=.94,bottom=.12,wspace=.39,hspace=.42)
    ax=aa[0,0]
    for j,(_,s) in enumerate(r.iterrows()):
        c=COLORS[j%8];ax.plot([0,1],[s.regulatory_rank,s.health_rank],'-o',c=c,alpha=.8)
        ax.text(-.035,s.regulatory_rank,short(s['item']),ha='right',va='center',fontsize=9)
        ax.text(1.04,s.health_rank,short(s['item']),va='center',fontsize=9)
    ax.set(xlim=(-.1,1.65),ylim=(10,0),xticks=[0,1],xticklabels=['Regulatory basis','HQ-augmented basis'],yticks=[]);ax.spines[['left','bottom']].set_visible(False);label(ax,'a','Ranking migration (benchmark-eligible subset)')
    ax=aa[0,1];base=np.zeros(len(r))
    for col,w,color,lab in [('toxicity_score',.5,BLUE,'Toxicity'),('occurrence_score',.3,ORANGE,'Reported occurrence'),('restriction_score',.2,RED,'Limit restriction')]:
        v=r[col]*w/36*100;ax.barh(range(len(r)),v,left=base,color=color,label=lab);base+=v
    ax.set(yticks=range(len(r)),yticklabels=[short(x) for x in r.item],xlabel='Regulatory composite score (0–100)',xlim=(0,100));ax.invert_yaxis();ax.legend(frameon=False,fontsize=8,loc='lower right');label(ax,'b','Components of the regulatory score');grid(ax,'x')
    ax=aa[1,0];xx=np.arange(4)
    for j,basis in enumerate(['regulatory','health']):
        pp=pr[pr.basis.eq(basis)].set_index('product').reindex(PRODUCTS)
        ax.bar(xx+(j-.5)*.34,pp.product_composite,width=.32,color=[BLUE,ORANGE][j],label=basis.capitalize())
        for x,v in zip(xx+(j-.5)*.34,pp.product_composite):
            if pd.isna(v):ax.text(x,2,'NA',ha='center',fontsize=8,rotation=90)
    ax.set(xticks=xx,xticklabels=['Honey','Royal\njelly','Bee\npollen','Other\nproducts'],ylim=(0,82),ylabel='Product-priority composite');ax.legend(frameon=False,loc='upper center',ncol=2,fontsize=8)
    cov=pr[pr.basis.eq('regulatory')].set_index('product').reindex(PRODUCTS).coverage_fraction
    for x,v in zip(xx,cov):ax.text(x,70,f'Eligible: {v:.1%}',ha='center',fontsize=8)
    label(ax,'c','Product scores and evidence coverage');grid(ax)
    ax=aa[1,1];s=sen.sort_values('rank_median');yy=np.arange(len(s));ax.hlines(yy,s.rank_min,s.rank_max,color='#aaaaaa',lw=3);ax.scatter(s.rank_median,yy,c=BLUE,label='Median rank');ax.scatter(s.baseline_health_rank,yy,c=ORANGE,marker='x',label='Baseline rank')
    ax.set(yticks=yy,yticklabels=[short(x) for x in s.item],xlabel='Rank across 473 weighting combinations',xlim=(.5,9.5));ax.invert_yaxis();ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(frameon=False,loc='lower right',fontsize=8);label(ax,'d','Weight sensitivity');grid(ax,'x')
    fig.text(.5,.035,'37 of 46 items lack an applicable verified ADI and remain ungraded, not low risk. Scores are decision conventions, not validated clinical risk scales.',ha='center',fontsize=9)
    savefig(fig,'Figure_6',['risk_ranking.csv','product_risk_ranking.csv','risk_weight_sensitivity.csv'],'Ranking migration, component scores, product composites with eligibility fractions, and all 473 weight-combination results.')

SC_COL={'S1':BLUE,'S2':'#17a2a4','S3':RED,'S4':'#888888','S5':ORANGE,'S6':GREEN,'S7':PURPLE}
SC_LABEL={'S1':'All locations','S2':'Available locations','S3':'Other / unknown','S4':'Norfloxacin only','S5':'By product','S6':'By stage','S7':'By region'}
def fig7():
    c=read('optimization_curves');k=read('optimization_knees');fig,aa=plt.subplots(2,2,figsize=(15,10));fig.subplots_adjust(left=.09,right=.97,top=.945,bottom=.10,hspace=.43,wspace=.30)
    ax=aa[0,0]
    for sc,g in c.groupby('scenario'):ax.plot(g.budget,g.coverage,label=sc+' '+SC_LABEL[sc],c=SC_COL[sc],ls='--' if sc=='S4' else '-',marker='o',markevery=10,ms=3)
    ax.set(xlabel='Budget (batch–item assessments)',ylabel='Modelled footprint coverage',ylim=(0,.9));pct(ax);ax.legend(frameon=False,fontsize=8,ncol=2);label(ax,'a','All seven scenarios; 101 budget points');grid(ax)
    ax=aa[0,1];s=c[c.budget.eq(15000)].sort_values('coverage',ascending=False);ax.bar(s.scenario,s.coverage,color=[SC_COL[x] for x in s.scenario]);pct(ax);ax.set(ylabel='Modelled footprint coverage',ylim=(0,.9))
    for j,v in enumerate(s.coverage):ax.text(j,v+.018,f'{v:.1%}',ha='center',fontsize=9)
    label(ax,'b','Scenario comparison at 15,000 assessments');grid(ax)
    ax=aa[1,0];sc=['S1','S6','S5','S7'];ss=c[c.budget.eq(15000)].set_index('scenario').loc[sc];xx=[2,3,4,8];ax.plot(xx,ss.coverage,'-o',color=BLUE)
    for x,code,v in zip(xx,sc,ss.coverage):ax.annotate(SC_LABEL[code]+f'\n{v:.1%}',(x,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.set(xlabel='Number of strata',ylabel='Modelled footprint coverage',ylim=(0,.95),xlim=(1,9),xticks=range(2,9));pct(ax);label(ax,'c','Stratification-scale comparison');grid(ax)
    ax=aa[1,1];s=k.set_index('scenario').loc[c[c.budget.eq(15000)].sort_values('coverage',ascending=False).scenario];yy=np.arange(len(s));ax.barh(yy,s.coverage_at_30000,color='#dddddd');ax.barh(yy,s.coverage_at_knee,color=[SC_COL[x] for x in s.index]);ax.set(yticks=yy,yticklabels=s.index,xlabel='Coverage at knee / 30,000 assessments',xlim=(0,1.15));ax.invert_yaxis();pct(ax,'x')
    for j,(_,v) in enumerate(s.iterrows()):ax.text(v.coverage_at_30000+.015,j,f'knee {v.knee_budget:,.0f}',va='center',fontsize=9)
    label(ax,'d','Knee and upper-budget comparison');grid(ax,'x')
    fig.text(.5,.025,'Coverage is a mathematical response to historical event weights; it does not estimate positive-test yield or demonstrated health benefit.',ha='center',fontsize=9)
    savefig(fig,'Figure_7',['optimization_curves.csv','optimization_knees.csv'],'Four optimization panels covering seven scenarios, fixed-budget comparison, scale and budget knees.')

def fig8():
    f=read('federated_comparison');dp=read('federated_dp_summary');cl=read('federated_clients');co=read('federated_communication_scaling')
    fig,aa=plt.subplots(2,2,figsize=(15,9.7));fig.subplots_adjust(left=.08,right=.96,top=.94,bottom=.12,hspace=.48,wspace=.32)
    ax=aa[0,0];s=f.iloc[:4].copy();dd=f[f.epsilon.eq(1)].iloc[0];s=pd.concat([s,pd.DataFrame([dd])],ignore_index=True)
    names=['Centralized','Federated\ncounts','Pooled +\nsplit budgets','Local +\nsplit budgets','Central DP\nε = 1'];ax.bar(range(5),s.coverage,color=['#888888',BLUE,'#a2b7c9',RED,'#17a2a4']);ax.set(xticks=range(5),xticklabels=names,ylabel='Footprint coverage',ylim=(0,.94));pct(ax)
    for j,v in enumerate(s.coverage):ax.text(j,v+.015,f'{v:.2%}',ha='center',fontsize=9)
    label(ax,'a','Federated utility at 30,000 assessments');grid(ax)
    ax=aa[0,1];ax.plot(dp.epsilon,dp.coverage_mean,'-o',color=BLUE,label='Coverage (mean ± SD)');ax.fill_between(dp.epsilon,dp.coverage_mean-dp.coverage_sd,dp.coverage_mean+dp.coverage_sd,color=BLUE,alpha=.17)
    ax.plot(dp.epsilon,dp.overlap_mean,'-s',c=ORANGE,label='Allocation overlap');ax.axhline(f.iloc[0].coverage,color='#777777',ls='--',lw=1,label='Centralized coverage');ax.set(xscale='log',xlabel='Privacy budget ε (smaller is stricter)',ylabel='Fraction',ylim=(0,1));pct(ax);ax.legend(frameon=False,fontsize=8);label(ax,'b','Central-DP privacy–utility curve');grid(ax)
    ax=aa[1,0]
    for mode,marker,col in [('Local BDeu','o',RED),('Uniform stratum prior','^','#999999')]:
        s=cl[cl['mode'].eq(mode)];ax.scatter(s.event_rows,s.pt_mean_L1,marker=marker,color=col,edgecolor='#555555',s=45,label=mode)
    for _,r in cl.nlargest(4,'event_rows').iterrows():ax.annotate(r.client,(r.event_rows,r.pt_mean_L1),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.axhline(0,c=BLUE,label='Federated = centralized');ax.set(xscale='log',xlabel='Client item-event rows',ylabel='Mean L1 deviation of stratum probabilities');ax.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,.10));label(ax,'c','Client size and local parameter deviation');grid(ax)
    ax=aa[1,1];ax.plot(co.event_rows,co.raw_bytes/1024,'-o',c=RED,label='Encoded raw records');ax.plot(co.event_rows,co.dense_upload_bytes/1024,'-s',c=BLUE,label='Dense upload: all 31 clients');ax.set(xscale='log',yscale='log',xlabel='Item-event rows (fixed client count)',ylabel='Payload (KiB; log scale)');ax.legend(frameon=False,fontsize=8);label(ax,'d','Communication and scale');grid(ax,'both')
    comm=read('federated_communication');ratio=comm.iloc[0].bytes/comm.iloc[1].bytes;ax.text(.05,.70,f'Observed raw / dense = {ratio:.3f}×\nDense counts are larger at the observed scale.',transform=ax.transAxes,fontsize=9,bbox={'fc':'white','ec':'none','alpha':.9})
    fig.text(.5,.025,'Fixed protocol; add/remove privacy unit = one report capped at five items; 20 noise repetitions. Public inputs and structure are outside this DP experiment.',ha='center',fontsize=9)
    savefig(fig,'Figure_8',['federated_comparison.csv','federated_dp_summary.csv','federated_clients.csv','federated_communication_scaling.csv'],'Federation utility, central-DP sensitivity, client heterogeneity and communication volume summed over all clients.')

def supplements():
    # S1: exposure of the drugs without an applicable ADI, lead MOE, ADI coverage and a matched federation control.
    e=read('exposure');moe=read('lead_moe');fig,aa=plt.subplots(2,2,figsize=(14.8,9.5));fig.subplots_adjust(left=.18,right=.97,top=.94,bottom=.12,wspace=.35,hspace=.45)
    ax=aa[0,0];s=e[e.adi_mgkg_bw_day.isna()&~e.item.eq('Lead')];s=s.loc[s.groupby('item').edi_baseline_ugkg_bw_day.idxmax()].sort_values('edi_baseline_ugkg_bw_day',ascending=False)
    ax.scatter(s.edi_baseline_ugkg_bw_day,range(len(s)),c=BLUE);ax.set(yticks=range(len(s)),yticklabels=[short(x) for x in s.item],xscale='log',xlabel='Conditional baseline EDI (µg/kg bw/day)');ax.invert_yaxis();label(ax,'a','Exposure retained where HQ is unavailable');grid(ax,'x')
    ax=aa[0,1];s=moe[moe.endpoint.eq('Adult kidney')];yy=np.arange(len(s));ax.hlines(yy,s.moe_sim_p05,s.moe_sim_median,color='#aaa',lw=3);ax.scatter(s.moe_sim_p05,yy,c=ORANGE,label='5th scenario percentile');ax.scatter(s.moe_sim_median,yy,c=BLUE,label='Median scenario');ax.axvline(1,color=RED,ls='--');ax.set(yticks=yy,yticklabels=[f'{x} (n={n})' for x,n in zip(s['product'],s.n)],xscale='log',xlabel='Lead MOE (adult kidney; pooled reporting period)');ax.legend(frameon=False,fontsize=8);label(ax,'b','Pooled-period contaminant assessment');grid(ax,'x')
    ax=aa[1,0];s=e[e.adi_mgkg_bw_day.notna()];tab=s.groupby('product').n.sum().reindex(PRODUCTS,fill_value=0);alln=e.groupby('product').n.sum().reindex(PRODUCTS,fill_value=0)
    ax.bar(range(4),alln,color='#ddd',label='All quantified chemicals');ax.bar(range(4),tab,color=GREEN,label='ADI-supported subset');ax.set(xticks=range(4),xticklabels=['Honey','Royal\njelly','Bee\npollen','Other'],ylabel='Quantified item events');ax.legend(frameon=False,fontsize=8);label(ax,'c','Coverage of health benchmarks');grid(ax)
    ax=aa[1,1];s=read('federated_minimum_policy_sensitivity');ss=s[s.budget.eq(15000)];ax.bar(range(3),ss.coverage,color=[BLUE,'#9ab4ca',RED]);ax.set(xticks=range(3),xticklabels=['Centralized','Pooled +\nsplit budgets','Local +\nsplit budgets'],ylabel='Footprint coverage',ylim=(0,.9));pct(ax);label(ax,'d','Shared national floor: 15,000 budget');grid(ax)
    savefig(fig,'Supplementary_Figure_S1',['exposure.csv','lead_moe.csv','federated_minimum_policy_sensitivity.csv'],'Baseline EDI of veterinary drugs without an applicable ADI, lead MOE, quantified events with and without an ADI, and split-budget allocations under one shared national minimum.')
    # S2: allocation, footprint contribution and marginal coverage across budgets.
    c=read('optimization_curves');alloc=read('optimization_allocations');con=read('optimization_stratum_contributions')
    fig,aa=plt.subplots(4,3,figsize=(15.5,15));fig.subplots_adjust(left=.075,right=.97,top=.97,bottom=.06,hspace=.56,wspace=.34)
    for row,sc in enumerate(['S1','S3','S5','S6']):
        g=alloc[alloc.scenario.eq(sc)].groupby(['budget','stratum']).allocated.sum().unstack(fill_value=0)
        ax=aa[row,0];ax.stackplot(g.index,g.T,labels=[short(x) for x in g.columns],colors=COLORS);ax.set(xlabel='Budget',ylabel='Allocated assessments');ax.legend(fontsize=7,frameon=False,loc='upper left');label(ax,string.ascii_lowercase[row*3],sc+' '+SC_LABEL[sc])
        g=con[con.scenario.eq(sc)].pivot(index='budget',columns='stratum',values='contribution');ax=aa[row,1]
        for j,col in enumerate(g):ax.plot(g.index,g[col],color=COLORS[j%8],label=short(col))
        ax.set(xlabel='Budget',ylabel='Footprint contribution');pct(ax);label(ax,string.ascii_lowercase[row*3+1]);grid(ax)
        g=c[c.scenario.eq(sc)];ax=aa[row,2];ax.plot(g.budget,g.marginal_per_1000*100,c=SC_COL[sc]);ax.axhline(1,c=RED,ls='--',lw=1);ax.set(xlabel='Budget',ylabel='Marginal coverage (pp / 1,000)');label(ax,string.ascii_lowercase[row*3+2]);grid(ax)
    savefig(fig,'Supplementary_Figure_S2',['optimization_allocations.csv','optimization_curves.csv','optimization_stratum_contributions.csv'],'Twelve panels: allocation, footprint contribution and marginal coverage in scenarios S1, S3, S5 and S6.')
    # S3: minimum stratum allocations.
    floors=read('optimization_floors');fa=read('optimization_floor_allocations');fig,aa=plt.subplots(2,2,figsize=(15.5,10));fig.subplots_adjust(left=.1,right=.97,top=.94,bottom=.13,hspace=.40,wspace=.36)
    top=read('item_summary').sort_values('records_with_item',ascending=False).head(10).item_en
    for j,layer in enumerate(['ProdR','Product']):
        ax=aa[0,j];s=fa[fa.layer.eq(layer)&fa.floor.eq(.2)];tab=s.pivot(index='item',columns='stratum',values='allocated').reindex(top)
        im=ax.imshow(np.log10(tab+1),aspect='auto',cmap='YlOrBr');ax.set(yticks=range(len(tab)),yticklabels=[short(x) for x in tab.index],xticks=range(len(tab.columns)),xticklabels=[short(x) for x in tab.columns]);ax.tick_params(axis='x',rotation=45);fig.colorbar(im,ax=ax,fraction=.035,label='log10(assessments + 1)');label(ax,'ab'[j],layer+' allocation, 20% floor (top 10 items)')
    ax=aa[1,0]
    for j,(layer,g) in enumerate(floors.groupby('layer')):ax.plot(g.floor,g.coverage,'-o',label=layer,c=COLORS[j])
    pct(ax);pct(ax,'x');ax.set(xlabel='National budget reserved across strata',ylabel='Footprint coverage');ax.legend(frameon=False);label(ax,'c','Cost of minimum stratum allocations');grid(ax)
    ax=aa[1,1]
    for j,(layer,g) in enumerate(floors.groupby('layer')):ax.plot(g.floor,g.min_per_stratum,'-o',label=layer,c=COLORS[j])
    ax.set(xlabel='National budget reserved across strata',ylabel='Minimum assessments in any stratum');pct(ax,'x');ax.legend(frameon=False);label(ax,'d','Minimum achieved allocation');grid(ax)
    savefig(fig,'Supplementary_Figure_S3',['optimization_floors.csv','optimization_floor_allocations.csv','item_summary.csv'],'Minimum stratum allocations at 15,000 assessments. Heatmaps show the ten most frequently reported items; complete allocations are in optimization_floor_allocations.csv.')
    # S4: footprint and HQ-weighted objectives.
    p=read('optimization_pareto');pa=read('optimization_pareto_allocations');rs=read('optimization_health_reserves');fig,aa=plt.subplots(2,2,figsize=(15.5,10));fig.subplots_adjust(left=.14,right=.97,top=.94,bottom=.11,hspace=.43,wspace=.35)
    ax=aa[0,0];t=pa[pa.alpha.isin([0,1])].groupby(['item','alpha']).allocated.sum().unstack(fill_value=0);t=t.loc[t.sum(axis=1).nlargest(12).index]
    yy=np.arange(len(t))
    ax.barh(yy-.17,t[0.],.32,color=BLUE,label='Footprint objective');ax.barh(yy+.17,t[1.],.32,color=ORANGE,label='HQ-weighted objective');ax.set(yticks=yy,yticklabels=[short(x) for x in t.index],xlabel='Allocated assessments');ax.invert_yaxis();ax.legend(frameon=False,fontsize=8);label(ax,'a','Objective-specific allocation (top 12)')
    ax=aa[0,1];ax.plot(p.coverage,p.health_fraction,'-o',c=PURPLE);pct(ax);pct(ax,'x');ax.set(xlabel='Footprint coverage',ylabel='Fraction of attainable screening mass');label(ax,'b','Weighted-objective Pareto frontier');grid(ax)
    ax=aa[1,0];ax.plot(rs.reserved_share,rs.coverage,'-o',color=BLUE,label='Footprint coverage');ax.plot(rs.reserved_share,rs.health_fraction,'-s',color=ORANGE,label='HQ-weighted coverage fraction');ax.set(xlabel='Budget reserve for top three median-HQ items',ylabel='Fraction');pct(ax);pct(ax,'x');ax.legend(frameon=False,fontsize=8);label(ax,'c','High-HQ-item reserve sensitivity');grid(ax)
    ax=aa[1,1];maxh=p.health_fraction.max();vals=[p.iloc[0].health_fraction/maxh,rs.iloc[2].health_fraction/maxh,rs.iloc[5].health_fraction/maxh,1.]
    ax.bar(['Footprint\nobjective','20% HQ\nreserve','50% HQ\nreserve','HQ-weighted\nobjective'],vals,color=[BLUE,GREEN,PURPLE,ORANGE]);ax.set(ylabel='Relative modelled HQ-weighted benefit',ylim=(0,1.15));pct(ax);label(ax,'d','Relative screening-priority coverage');grid(ax)
    savefig(fig,'Supplementary_Figure_S4',['optimization_pareto.csv','optimization_pareto_allocations.csv','optimization_health_reserves.csv'],'Footprint and HQ-weighted objectives: item-level allocation, Pareto mixtures and reserves for the three items with the largest median HQ; the HQ-weighted objective comprises the nine chemicals with an applicable ADI.')

def graphical_abstract():
    fig=plt.figure(figsize=(14,7.4));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.5,.96,'Integrated bee-product surveillance, exposure screening and monitoring design',ha='center',fontsize=17,fontweight='bold',fontfamily='DejaVu Serif')
    def box(x,y,w,h,color,title):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.005,rounding_size=.014',fc='#fcfcfc',ec='#666',lw=1.1))
        ax.add_patch(Rectangle((x,y+h-.065),w,.065,fc=color,ec='none'))
        ax.text(x+w/2,y+h-.033,title,ha='center',va='center',fontweight='bold',fontsize=12)
    box(.025,.075,.19,.82,'#dddddd','Data sources')
    box(.255,.64,.435,.255,'#cfe3f3','Hazard profiling and Bayesian spatial inference')
    box(.255,.355,.435,.255,'#d9edd3','Direct-consumption exposure and priority')
    box(.255,.075,.435,.25,'#fce9b8','All categories retained; limitations quantified')
    box(.735,.495,.24,.4,'#e4daf0','Monitoring optimization')
    box(.735,.075,.24,.39,'#e4daf0','Federated statistics')
    ax.text(.12,.735,'87,313',ha='center',fontsize=26,fontweight='bold',color=BLUE)
    ax.text(.12,.68,'monitored batches',ha='center',fontsize=11)
    ax.text(.12,.55,'1,798',ha='center',fontsize=26,fontweight='bold',color=RED)
    ax.text(.12,.495,'noncompliant reports',ha='center',fontsize=10.5)
    ax.text(.12,.38,'4 products\n46 indicators',ha='center',fontsize=17,fontweight='bold',linespacing=1.7)
    ax.text(.12,.19,'China regulatory reports\n2015–20 September 2026\nObserved failure fraction: 2.06%',ha='center',fontsize=9,linespacing=1.6)
    for yy in [.735,.46,.20]:
        ax.annotate('',(.25,yy),(.22,yy),arrowprops={'arrowstyle':'-|>','mutation_scale':20,'color':'#777','lw':2})
        ax.annotate('',(.73,yy),(.695,yy),arrowprops={'arrowstyle':'-|>','mutation_scale':20,'color':'#777','lw':2})
    # Mini-panels use actual model output, not stock or generative imagery.
    mini=fig.add_axes([.285,.674,.15,.13]);p=read('spatial_footprint').pivot(index='production_region',columns='sampling_region',values='footprint_share');mini.imshow(p,cmap='Blues',aspect='auto');mini.set_xticks([]);mini.set_yticks([]);mini.set_title('Production–sampling footprint',fontsize=8)
    ax.text(.555,.743,'Six-node TAN\nGrouped validation\nUnknown locations retained',ha='center',fontsize=11,linespacing=1.5)
    ax.text(.475,.52,'EDI = C × DI / (1000 × BW)',ha='center',va='center',fontsize=14,fontweight='bold',color='#365b35')
    ax.text(.475,.445,'Veterinary residues: conditional EDI\n9 ADI-supported chemicals: HQ\nConsumption and weight sensitivity',ha='center',va='center',fontsize=11,linespacing=1.65)
    ax.text(.475,.374,'Product-specific direct-consumption scenarios',ha='center',fontsize=10,color='#365b35')
    ax.text(.475,.177,'Seven allocation scenarios • 473 ranking-weight combinations\nBenchmark availability is reported explicitly\nConditional screening and monitoring simulations',ha='center',va='center',fontsize=10,linespacing=1.8)
    mini=fig.add_axes([.776,.635,.17,.16]);c=read('optimization_curves')
    for sc in ['S1','S5','S7']:
        g=c[c.scenario.eq(sc)];mini.plot(g.budget/1000,g.coverage*100,c=SC_COL[sc],label=sc)
    mini.set(xlabel='Budget (thousands)',ylabel='Footprint coverage (%)');mini.tick_params(labelsize=7);mini.xaxis.label.set_size(8);mini.yaxis.label.set_size(8);mini.legend(frameon=False,fontsize=7,loc='lower right');grid(mini)
    ax.text(.855,.535,'Integer budgets • Coverage floors\nPareto and exposure-priority scenarios',ha='center',va='center',fontsize=9)
    f=read('federated_comparison');ax.text(.855,.35,'Count aggregation = pooled model',ha='center',fontsize=10,fontweight='bold',color=BLUE)
    mini=fig.add_axes([.777,.19,.17,.13]);dp=read('federated_dp_summary');mini.plot(dp.epsilon,dp.coverage_mean*100,'-o',ms=3,c=BLUE);mini.set(xscale='log',xlabel='Privacy budget ε',ylabel='Coverage (%)');mini.tick_params(labelsize=7);mini.xaxis.label.set_size(8);mini.yaxis.label.set_size(8);grid(mini)
    ax.text(.855,.10,'Central-DP simulation; all-client payload',ha='center',fontsize=8)
    ax.text(.5,.026,'Retrospective surveillance records • Conditional exposure screening • Explicit monitoring-design scenarios',ha='center',fontsize=9,color='#555')
    savefig(fig,'Graphical_Abstract',['summary.json','spatial_footprint.csv','optimization_curves.csv','federated_dp_summary.csv'],'Three-column graphical abstract: data sources, analytical modules, and monitoring optimization with federated statistics. The embedded charts are drawn from the result files.')

def main():
    for fn in [fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,supplements,graphical_abstract]:fn()
    (FIG/'figure_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
