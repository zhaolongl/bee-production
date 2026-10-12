"""Reconciliation checks and descriptive summaries from the tables in data/.

All counts describe the compiled regulatory reports. Proportions use monitored
batches as denominators; no item-specific test denominator is inferred.
"""
import importlib.metadata
import platform
from bee_common import *

TABLES = ['records','items','monthly','province','authority','reported_reasons',
          'product_denominators','product_monthly','product_province','product_authority']
GRAND = [87313,85515,1798]
CHEMICAL = ['兽药','农药','污染物','食品添加剂']
PACKAGES = ['numpy','pandas','scipy','scikit-learn','statsmodels','patsy','matplotlib','Pillow']

def tables():
    return {n:pd.read_csv(DATA/(n+'.csv'),float_precision='round_trip') for n in TABLES}

def reconcile(t):
    """Check the aggregate tables against one another and against the detailed records."""
    records,items,monthly,province,authority,reasons,totals,months,provinces,authorities=(t[n] for n in TABLES)
    checks=[]
    for name,frame in [('monthly',monthly),('production_province',province),('authority',authority)]:
        assert (frame.total==frame.passed+frame.failed).all(),name
        assert frame[['total','passed','failed']].sum().tolist()==GRAND,name
        checks.append({'check':f'{name}_sum_and_arithmetic','status':'PASS','detail':'87313 = 85515 + 1798'})
    for label,df in [('product_total',totals),('product_month',months),('product_province',provinces),('product_authority',authorities)]:
        assert (df.total==df.passed+df.failed).all(),label
        for row in totals.itertuples():
            assert df.loc[df.product_en.eq(row.product_en),['total','passed','failed']].sum().tolist()==[row.total,row.passed,row.failed],(label,row.product_en)
        checks.append(dict(check=label+'_PDF_arithmetic',status='PASS',detail=f'{len(df)} rows; each subtype total matches'))
    assert totals[['total','passed','failed']].sum().tolist()==GRAND
    assert len(months)==804 and not months.duplicated(['product_en','date']).any()
    msum=months.groupby(['year','month'])[['total','passed','failed']].sum()
    old_month=monthly.set_index(['year','month'])[['total','passed','failed']].sort_index()
    assert msum.index.equals(old_month.index) and np.array_equal(msum.to_numpy(),old_month.to_numpy())
    psum=provinces.groupby('province_cn')[['total','passed','failed']].sum().sort_index()
    old_province=province.set_index('province_cn')[['total','passed','failed']].sort_index()
    assert psum.index.equals(old_province.index) and np.array_equal(psum.to_numpy(),old_province.to_numpy())
    detail=records.groupby(['product_en','year','month']).size()
    assert all(detail.get((r.product_en,r.year,r.month),0)==r.failed for r in months.itertuples())
    detail=records.groupby(['product_en','production_province']).size()
    assert all(detail.get((r.product_en,r.province_cn),0)==r.failed for r in provinces.itertuples())
    checks.append(dict(check='subtype_margins_and_detail_reconciliation',status='PASS',detail='All product-month and product-province failures reconcile; subtype sums match all-product monthly and province totals'))
    # Validate against all available marginal denominators, including the zero months.
    for label,detail_col,agg_col,agg in [('production_province','production_province','province_cn',province),
                                         ('authority','source_authority','authority',authority)]:
        counts=records[detail_col].value_counts().reindex(agg[agg_col]).fillna(0).to_numpy()
        assert np.array_equal(counts,agg.failed.to_numpy()),label
        checks.append({'check':label+'_detail_reconciliation','status':'PASS','detail':str(len(agg))+' strata'})
    mc=records.groupby(['year','month']).size()
    assert all(mc.get((r.year,r.month),0)==r.failed for r in monthly.itertuples())
    checks.append({'check':'monthly_detail_reconciliation','status':'PASS','detail':str(len(monthly))+' months'})
    reason_counts=items.item_cn.value_counts().reindex(reasons.item_cn).fillna(0).to_numpy()
    assert np.array_equal(reason_counts,reasons.item_entries.to_numpy())
    checks.append({'check':'all_reason_entries_reconciled','status':'PASS','detail':f'{len(items)} entries / {len(reasons)} items'})
    assert len(records)==1798 and len(items)==1922
    return checks

def main():
    t=tables();checks=reconcile(t)
    records,items,monthly=t['records'],t['items'],t['monthly']
    product_totals,product_monthly=t['product_denominators'],t['product_monthly']
    annual=add_rate(monthly.groupby('year')[['total','passed','failed']].sum().reset_index())
    annual['partial_year']=annual.year==2026
    annual=annual[annual.total>0]
    months_complete=monthly[(monthly.year.isin([2025,2026]))&(monthly.month<=8)]
    ytd=add_rate(months_complete.groupby('year')[['total','passed','failed']].sum().reset_index())
    product_annual=add_rate(product_monthly.groupby(['product_en','year'])[['total','passed','failed']].sum().reset_index())
    unique_items=items.drop_duplicates(['record_id','item_cn'])
    category_record=items.drop_duplicates(['record_id','category_en'])
    cs=category_record.groupby('category_en').size().rename('records_with_category').reset_index()
    cs['share_of_failed_records']=cs.records_with_category/len(records)
    cs=cs.merge(items.groupby('category_en').size().rename('item_entries'),on='category_en')
    cs=cs.sort_values('records_with_category',ascending=False)
    it=items.groupby(['item_cn','item_en','category_en']).size().rename('item_entries').reset_index()
    it=it.merge(unique_items.groupby('item_cn').size().rename('records_with_item'),on='item_cn')
    it['share_of_failed_records']=it.records_with_item/len(records)
    it=it.sort_values('records_with_item',ascending=False)
    assert not it.item_cn.duplicated().any(), 'Each standardized item must have one canonical category'
    yc=category_record.groupby(['year','category_en']).size().rename('records_with_category').reset_index()
    yc=yc.merge(records.groupby('year').size().rename('failed_records'),on='year')
    yc['share_of_failed_records']=yc.records_with_category/yc.failed_records
    pc=category_record.groupby(['product_en','category_en']).size().rename('records_with_category').reset_index()
    pc=pc.merge(records.groupby('product_en').size().rename('failed_records'),on='product_en')
    pc['share_of_failed_records']=pc.records_with_category/pc.failed_records
    prod=records.groupby('product_en').agg(failed_records=('record_id','size'),veterinary_records=('has_veterinary','sum'),chemical_records=('has_chemical','sum')).reset_index()
    prod=prod.merge(product_totals,on='product_en')
    flow=records.groupby(['production_region','sampling_region']).size().rename('records').reset_index()
    valid_flow=records.production_region.ne('Other / unknown') & records.sampling_region.ne('Unknown')
    # Retain every unresolved value and candidate duplicate for source review.
    chemical=items.category_cn.isin(CHEMICAL)
    quant=items[items.quantitative_primary].copy()
    conc=quant.groupby(['item_en','category_en']).concentration_ugkg.agg(n='size',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),minimum='min',maximum='max').reset_index()
    qa=items[chemical].groupby(['category_en','parse_status']).size().rename('item_entries').reset_index()
    # Reports whose value or limit list could not be matched one-to-one with the item list.
    bad=items[~(items.value_alignment_ok & items.limit_alignment_ok)]
    parts=lambda s:s.fillna('').astype(str).str.split('|').str.len()
    alignment=pd.DataFrame({'record_id':bad.record_id,'n_items':bad.groupby('record_id').item_order.transform('size'),
        'n_values':parts(bad.values_full),'n_limits':parts(bad.limits_full),'value_alignment_ok':bad.value_alignment_ok,
        'limit_alignment_ok':bad.limit_alignment_ok,'source_row':bad.source_row}).drop_duplicates('record_id')
    # Sensitivity only: collapse identical rows without pretending they are proven duplicate batches.
    sens_records=records.loc[~records.exact_duplicate_of_earlier_row]
    sensitivity=pd.DataFrame([
       {'analysis':'Primary: all platform records','n_failed_records':len(records),'veterinary_records':int(records.has_veterinary.sum()),'interpretation':'Platform records; matches all marginal denominators'},
       {'analysis':'Sensitivity: collapse identical records','n_failed_records':len(sens_records),'veterinary_records':int(sens_records.has_veterinary.sum()),'interpretation':'Not a replacement numerator for the platform denominator'}])
    summary={
       'total':GRAND[0],'passed':GRAND[1],'failed_records':len(records),'overall_rate':GRAND[2]/GRAND[0],
       'overall_ci_low':float(wilson(GRAND[2],GRAND[0])[1]),'overall_ci_high':float(wilson(GRAND[2],GRAND[0])[2]),
       'first_notice':records.date.min(),'last_notice':records.date.max(),'extract_end':'2026-09-20',
       'item_entries':len(items),'unique_record_item_pairs':len(unique_items),'canonical_items':items.item_cn.nunique(),
       'multi_item_records':int((records.item_count>1).sum()),'veterinary_records':int(records.has_veterinary.sum()),
       'chemical_records':int(records.has_chemical.sum()),'unknown_origin_records':int(records.production_province.eq('不详').sum()),
       'sampling_province_known_records':int(records.sampling_province.ne('不详').sum()),
       'known_domestic_region_pairs':int(valid_flow.sum()),
       'cross_region_pairs':int((valid_flow & records.production_region.ne(records.sampling_region)).sum()),
       'same_region_pairs':int((valid_flow & records.production_region.eq(records.sampling_region)).sum()),
       'identical_row_excess':int(records.exact_duplicate_of_earlier_row.sum()),'records_in_identical_groups':int(records.possible_exact_duplicate.sum()),
       'chemical_item_entries':int(chemical.sum()),'quantified_chemical_entries':len(quant),
       'category_corrections':int(items.category_corrected.sum()),
       'unit_source_check_entries':int(items.parse_status.eq('unit_requires_source_check').sum()),
       'validation_checks':checks,'seed':SEED}
    for name,frame in [('annual',annual),('jan_aug_comparison',ytd),('product_annual',product_annual),('category_summary',cs),('item_summary',it),
                       ('year_category',yc),('product_category',pc),('product_summary',prod),('region_pairs',flow),('concentrations',quant),
                       ('concentration_summary',conc),('quantification_audit',qa),('alignment_review',alignment),
                       ('chemical_value_review',items[chemical & ~items.quantitative_primary]),('category_corrections',items[items.category_corrected]),
                       ('possible_duplicate_records',records[records.possible_exact_duplicate]),('duplicate_sensitivity',sensitivity),
                       ('validation_checks',pd.DataFrame(checks))]:save(frame,name)
    jsave(summary,'summary')
    jsave({'python':platform.python_version(),'packages':{m:importlib.metadata.version(m) for m in PACKAGES},'seed':SEED},'run_manifest')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
