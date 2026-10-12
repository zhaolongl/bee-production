"""Build the analysis tables in data/ from the raw platform exports.

The exports (two workbooks and four product reports in PDF) are read-only inputs
located in RAW (see bee_common). They are not distributed with the repository;
the tables written here are, so every later step runs without them.
No test denominators are inferred and no record is removed.
"""
import hashlib
import re
import unicodedata
from datetime import datetime

from bee_common import *

CATEGORY = {'兽药':'Veterinary drugs','微生物':'Microbiology','质量指标':'Quality indicators',
            '食品添加剂':'Food additives','污染物':'Lead','农药':'Pesticides','标签':'Labelling','其他类':'Other'}
PRODUCT = {'蜂蜜':'Honey','其他蜂产品':'Other bee products','蜂花粉':'Bee pollen','蜂王浆':'Royal jelly'}
PRODUCT_REPORT = {'蜂蜜':'蜂蜜','蜂王浆':'蜂王浆','蜂花粉':'蜂花粉','其他蜂产品':'其它蜂产品'}
WORKBOOK_ALL, WORKBOOK_FAILED = '所有数据.xlsx', '蜂产品 不合格数据.xlsx'
ALIASES = {'细菌总数（菌落总数）':'菌落总数','标签及标志':'标签标识','铅':'铅（以Pb计）',
 '山梨酸及其盐':'山梨酸及其钾盐(以山梨酸计)','苯甲酸及其盐':'苯甲酸及其钠盐(以苯甲酸计)',
 '环己基氨基磺酸钠(甜蜜素)':'甜蜜素（以环己基氨基磺酸计）','乙酰磺胺酸钾(安赛蜜)':'安赛蜜',
 '磺胺类':'磺胺类（总量）','总糖':'总糖分'}
ITEM_EN = {'菌落总数':'Aerobic plate count','果糖和葡萄糖':'Fructose + glucose','诺氟沙星':'Norfloxacin',
 '嗜渗酵母计数':'Osmophilic yeasts','呋喃西林代谢物':'Nitrofurazone metabolite (SEM)',
 '氯霉素':'Chloramphenicol','霉菌':'Moulds','10-羟基-2-癸烯酸':'10-HDA','甲硝唑':'Metronidazole',
 '蔗糖':'Sucrose','包装空隙率':'Packaging void ratio','山梨酸及其钾盐(以山梨酸计)':'Sorbates',
 '标签标识':'Labelling','甜蜜素（以环己基氨基磺酸计）':'Cyclamate',
 '苯甲酸及其钠盐(以苯甲酸计)':'Benzoates','硝基咪唑类':'Nitroimidazoles',
 '铅（以Pb计）':'Lead','氧氟沙星':'Ofloxacin','氟喹诺酮类':'Fluoroquinolones',
 '净含量标示':'Net-content declaration','安赛蜜':'Acesulfame K','糖精钠':'Sodium saccharin',
 '呋喃唑酮代谢物':'Furazolidone metabolite (AOZ)','包装要求':'Packaging requirements',
 '不详':'Unspecified','大肠菌群':'Coliforms','四环素类':'Tetracyclines','硝基呋喃类':'Nitrofurans',
 '培氟沙星':'Pefloxacin','磺胺类（总量）':'Sulfonamides','水分':'Moisture','酸度':'Acidity',
 '淀粉酶活性':'Diastase activity','还原糖':'Reducing sugars','双甲脒':'Amitraz',
 '日落黄及其铝色淀':'Sunset yellow','净含量':'Net content','氟胺氰菊酯':'Tau-fluvalinate',
 '柠檬黄及其铝色淀':'Tartrazine','羟甲基糠醛':'Hydroxymethylfurfural','总灰分':'Total ash',
 '营养成分表':'Nutrition declaration','配料表标识问题':'Ingredient declaration',
 '蛋白质':'Protein','总糖分':'Total sugars','淀粉':'Starch'}

def text(v):
    return '' if pd.isna(v) else str(v).strip()

def raw_files():
    return [RAW/WORKBOOK_FAILED, RAW/WORKBOOK_ALL, *[RAW/f'食品安全合规管理智慧平台{s}.pdf' for s in PRODUCT_REPORT.values()]]

def write_csv(frame, name):
    frame.to_csv(DATA/name, index=False, encoding='utf-8-sig')

def find_row(frame, title):
    return next(i for i,v in frame[0].items() if text(v)==title)

def aggregates():
    raw = pd.read_excel(RAW/WORKBOOK_ALL,header=None)
    monthly = raw[raw[0].map(lambda x:isinstance(x,datetime))].iloc[:,:4].copy()
    monthly.columns=['date','total','passed','failed']
    monthly['date']=pd.to_datetime(monthly['date'])
    monthly['year']=monthly.date.dt.year
    monthly['month']=monthly.date.dt.month
    for c in ['total','passed','failed']: monthly[c]=monthly[c].astype(int)
    def block(start,end,cols):
        a,b=find_row(raw,start)+2,find_row(raw,end)
        t=raw.iloc[a:b,:len(cols)].copy();t.columns=cols
        for c in ['total','passed','failed']:
            if c in t:t[c]=t[c].astype(int)
        return t
    province=block('4、生产省份统计','5、通报单位统计',['province_cn','total','passed','failed','reported_pass_rate'])
    authority=block('5、通报单位统计','6、合格率统计',['authority','total','passed','failed','reported_pass_rate'])
    reasons=block('3、不合格原因统计','4、生产省份统计',['product','category_cn','item_cn','item_entries','reported_share'])
    reasons[['product','category_cn']]=reasons[['product','category_cn']].ffill()
    reasons['item_entries']=reasons.item_entries.astype(int)
    # The report shortens the metabolite to the parent drug. Keep the analytical label.
    reasons['item_cn']=reasons.item_cn.replace({'呋喃西林':'呋喃西林代谢物'})
    province['province_en']=province.province_cn.map(PROVINCES)
    province['region']=province.province_cn.map(PROVINCE_REGION).fillna('Other / unknown')
    return monthly,province,authority,reasons

def sampling_province(address):
    # Only explicit province/municipality at the start; no city-name lookup or imputation.
    address=re.sub(r'^(被抽样单位地址|抽样单位地址)[:：]?','',text(address)).strip()
    for p in PROVINCES:
        if p not in ['不详','进口'] and address.startswith(p):return p
    return '不详'

def numeric_mass(v):
    """Strict single-result parser. No unit inference, censor substitution or repair."""
    s=text(v)
    if not s:return np.nan,'missing',''
    if re.search(r'未检出|不得|检出|超标|不合格',s):return np.nan,'qualitative',''
    s=re.sub(r'<sup>(.*?)</sup>',r'^\1',s)
    s=unicodedata.normalize('NFKC',s).replace('μ','u').replace('µ','u')
    s=re.sub(r'\s+','',s).replace('×','x')
    if s.startswith(('<','>','≤','≥')):return np.nan,'censored',''
    m=re.fullmatch(r'(\d+(?:\.\d+)?)(?:[x*]10\^?([+-]?\d+))?(ug/kg|mg/kg|g/kg|ng/g|ng/kg)',s,re.I)
    if not m:return np.nan,'unparsed_or_unit_missing',''
    value=float(m[1])*(10**int(m[2]) if m[2] else 1)
    unit=m[3].lower();value*= {'ug/kg':1,'mg/kg':1000,'g/kg':1e6,'ng/g':1,'ng/kg':.001}[unit]
    return value,'parsed',unit

def product_denominators():
    """Extract the four product reports, preserving PDF page provenance."""
    import pymupdf
    totals=[];months=[];provinces=[];authorities=[]
    for product,suffix in PRODUCT_REPORT.items():
        source=RAW/f'食品安全合规管理智慧平台{suffix}.pdf'
        doc=pymupdf.open(source)
        pages=[page.get_text() for page in doc]
        content='\n'.join(pages)
        match=re.search(r'共抽检(\d+)批次，其中合格(\d+)批次，不合格(\d+)批次',content)
        assert match,source.name
        total,passed,failed=map(int,match.groups())
        totals.append(dict(product_cn=product,product_en=PRODUCT[product],total=total,passed=passed,failed=failed,source_pdf=source.name,source_page=1))
        for page_no,page_text in enumerate(pages,1):
            for m in re.finditer(r'(20\d{2}-\d{2})\s*\n(\d+)\s*\n(\d+)\s*\n(\d+)',page_text):
                date,n,k0,k1=m.groups()
                months.append(dict(product_en=PRODUCT[product],date=date+'-01',year=int(date[:4]),month=int(date[5:]),total=int(n),passed=int(k0),failed=int(k1),source_pdf=source.name,source_page=page_no))
        block=content.split('4、生产省份统计',1)[1].split('5、通报单位统计',1)[0]
        for m in re.finditer(r'([^\n]+)\n(\d+)\n(\d+)\n(\d+)\n([\d.]+)%',block):
            name,n,k0,k1,rate=m.groups()
            if name.strip() not in PROVINCES: continue
            provinces.append(dict(product_en=PRODUCT[product],province_cn=name.strip(),province_en=PROVINCES[name.strip()],total=int(n),passed=int(k0),failed=int(k1),source_pdf=source.name))
        block=content.split('5、通报单位统计',1)[1].split('6、合格率统计',1)[0]
        for m in re.finditer(r'([^\n]+)\n(\d+)\n(\d+)\n(\d+)\n([\d.]+)%',block):
            name,n,k0,k1,rate=m.groups()
            if not any(token in name for token in ['监管','监督','农业','药监']):continue
            authorities.append(dict(product_en=PRODUCT[product],authority=name.strip(),total=int(n),passed=int(k0),failed=int(k1),source_pdf=source.name))
    return [pd.DataFrame(x) for x in (totals,months,provinces,authorities)]

def records_and_items():
    raw=pd.read_excel(RAW/WORKBOOK_FAILED).fillna('')
    records=[];items=[]
    raw_key_cols=[c for c in raw if c!='序号' and not c.startswith('Unnamed')]
    duplicate_flags=raw.duplicated(raw_key_cols,keep=False)
    # Second and later copies of an identical row; used only for the sensitivity count.
    repeat_flags=raw.duplicated(raw_key_cols)
    for idx,row in raw.iterrows():
        rid=f'B{idx+1:04d}';date=pd.to_datetime(row['通报时间'])
        names=[a.strip() for a in text(row['不合格规范列']).split(',')]
        source_cats=[a.strip() for a in text(row['不合格项目分类']).split(',')]
        # Two source classification errors are corrected by explicit analyte semantics.
        category_overrides={'果糖和葡萄糖':'质量指标','嗜渗酵母计数':'微生物'}
        cats=[category_overrides.get(ALIASES.get(name,name),cat) for name,cat in zip(names,source_cats)]
        assert len(names)==len(cats),(rid,names,cats)
        # Source-notice grouping is deliberately conservative when batch identity is incomplete.
        notice=text(row['来源链接']) or text(row['伙伴网链接']) or '|'.join([str(date.date()),text(row['通报单位']),text(row['通报文号'])])
        group_parts=[text(row[c]) for c in ['产品名称','生产企业名称','生产日期','抽样单位名称']]
        group_key='|'.join(group_parts) if all(group_parts) else notice+'|'+text(row['产品名称'])
        province=text(row['生产省份']);sample_prov=sampling_province(row['抽样单位地址'])
        rec=dict(record_id=rid,source_row=idx+2,source_serial=row['序号'],date=date.date().isoformat(),
             year=date.year,month=date.month,product_cn=text(row['产品分类']).split('-->')[-1],
             production_province=province,production_region=PROVINCE_REGION.get(province,'Other / unknown'),
             sampling_province=sample_prov,sampling_region=PROVINCE_REGION.get(sample_prov,'Unknown'),
             sampling_level=text(row['抽检级别']),sampling_stage=text(row['抽样环节']),
             source_notice=notice,source_authority=text(row['通报单位']),item_count=len(names),
             source_link=text(row['来源链接']),
             group_id=hashlib.sha256(group_key.encode()).hexdigest()[:16],
             possible_exact_duplicate=bool(duplicate_flags.iloc[idx]),
             exact_duplicate_of_earlier_row=bool(repeat_flags.iloc[idx]),
             has_veterinary=int('兽药' in cats),has_microbiology=int('微生物' in cats),
             has_chemical=int(any(c in cats for c in ['兽药','农药','污染物','食品添加剂'])))
        rec['product_en']=PRODUCT[rec['product_cn']]
        records.append(rec)
        values=text(row['检测数值']).split('|');limits=text(row['法规限制']).split('|')
        val_ok=len(values)==len(names);lim_ok=len(limits)==len(names)
        for j,(name,cat) in enumerate(zip(names,cats)):
            canonical=ALIASES.get(name,name);assert canonical in ITEM_EN,canonical
            value=values[j] if val_ok else '';limit=limits[j] if lim_ok else ''
            mass,status,unit=numeric_mass(value)
            chemical=cat in ['兽药','农药','污染物','食品添加剂']
            if not chemical:mass=np.nan;status='not_chemical'
            elif not val_ok:status='alignment_unresolved'
            if cat=='兽药' and unit=='g/kg':status='unit_requires_source_check'
            item=dict(record_id=rid,item_order=j+1,year=date.year,product_en=rec['product_en'],
              category_cn=cat,category_en=CATEGORY[cat],source_category_cn=source_cats[j],category_corrected=cat!=source_cats[j],item_raw=name,item_cn=canonical,item_en=ITEM_EN[canonical],
              value_raw=value,limit_raw=limit,values_full=text(row['检测数值']),limits_full=text(row['法规限制']),
              value_alignment_ok=val_ok,limit_alignment_ok=lim_ok,concentration_ugkg=mass,
              parse_status=status,parsed_unit=unit,quantitative_primary=bool(chemical and status=='parsed' and mass>0),
              source_row=idx+2,source_serial=row['序号'],source_link=text(row['来源链接']))
            items.append(item)
    return pd.DataFrame(records),pd.DataFrame(items)

def main():
    missing=[p.name for p in raw_files() if not p.is_file()]
    if missing:raise SystemExit(f'Raw inputs not found in {RAW}: '+', '.join(missing))
    DATA.mkdir(parents=True,exist_ok=True)
    monthly,province,authority,reasons=aggregates()
    records,items=records_and_items()
    totals,months,provinces,authorities=product_denominators()
    assert len(records)==1798 and len(items)==1922
    assert not (items.quantitative_primary & items.concentration_ugkg.isna()).any()
    for name,frame in [('records.csv',records),('items.csv',items),('reported_reasons.csv',reasons)]:write_csv(frame,name)
    for name,frame in [('monthly.csv',monthly),('province.csv',province),('authority.csv',authority),('product_denominators.csv',totals),
                       ('product_monthly.csv',months),('product_province.csv',provinces),('product_authority.csv',authorities)]:write_csv(add_rate(frame),name)
    write_csv(pd.DataFrame([{'item_cn':k,'item_en':v} for k,v in ITEM_EN.items()]),'item_dictionary.csv')
    manifest={'inputs':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in raw_files()],
      'raw_inputs_modified':False,'deduplication_policy':'No automatic deletion. Collapsing identical rows is a sensitivity analysis only.'}
    (DATA/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # The written tables must reconcile exactly as they will for a user without the raw inputs.
    from bee_descriptive import tables,reconcile
    checks=reconcile(tables())
    print(f'Wrote {len(records)} records and {len(items)} item entries to {DATA}; {len(checks)} reconciliation checks passed.')

if __name__=='__main__':main()
