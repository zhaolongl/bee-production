"""Shared paths, constants and loaders for the bee-product surveillance analysis."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, FIG = ROOT/'data', ROOT/'results', ROOT/'figures'
# The raw platform exports are not distributed; bee_preprocess.py reads them from here.
RAW = Path(os.environ.get('BEE_RAW_DIR', ROOT/'raw'))
SEED, NT, ESS = 20260920, 87313, 5.0
PRODUCTS = ['Honey', 'Royal jelly', 'Bee pollen', 'Other bee products']
REGIONS = ['East','Central','South','Southwest','Northwest','North','Northeast','Other / unknown']
CATS = ['Veterinary drugs','Microbiology','Quality indicators','Food additives','Labelling','Other','Lead','Pesticides']
COLORS = ['#3b76af','#e1812c','#3a923a','#9467bd','#8c564b','#777777','#c95a5b','#17a2a4']
SHORT = {'Nitrofurazone metabolite (SEM)':'SEM','Furazolidone metabolite (AOZ)':'AOZ','Sodium saccharin':'Saccharin','Other bee products':'Other products','Other / unknown':'Other / unknown'}
# Province names as they appear in the source records, with English names and regions.
PROVINCES = dict(zip('北京 天津 河北 山西 内蒙古 辽宁 吉林 黑龙江 上海 江苏 浙江 安徽 福建 江西 山东 河南 湖北 湖南 广东 广西 海南 重庆 四川 贵州 云南 西藏 陕西 甘肃 青海 宁夏 新疆 台湾 香港 进口 不详'.split(),
                    'Beijing|Tianjin|Hebei|Shanxi|Inner Mongolia|Liaoning|Jilin|Heilongjiang|Shanghai|Jiangsu|Zhejiang|Anhui|Fujian|Jiangxi|Shandong|Henan|Hubei|Hunan|Guangdong|Guangxi|Hainan|Chongqing|Sichuan|Guizhou|Yunnan|Tibet|Shaanxi|Gansu|Qinghai|Ningxia|Xinjiang|Taiwan|Hong Kong|Imported|Unknown'.split('|')))
PROVINCE_REGION = {p:region for region,members in {'East':'上海 江苏 浙江 安徽 福建 江西 山东','North':'北京 天津 河北 山西 内蒙古',
                   'Northeast':'辽宁 吉林 黑龙江','Central':'河南 湖北 湖南','South':'广东 广西 海南',
                   'Southwest':'重庆 四川 贵州 云南 西藏','Northwest':'陕西 甘肃 青海 宁夏 新疆'}.items() for p in members.split()}

def save(df, name):
    OUT.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT/(name+'.csv'), index=False, encoding='utf-8-sig')
    return df

def jsave(obj, name):
    (OUT/(name+'.json')).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)), encoding='utf-8')

def wilson(k, n):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k/n
        z = norm.ppf(.975)
        center = (p + z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return p, np.where(n>0, center-half, np.nan), np.where(n>0, center+half, np.nan)

def add_rate(df):
    df = df.copy()
    df['rate'], df['ci_low'], df['ci_high'] = wilson(df['failed'],df['total'])
    return df

def load_data():
    r = pd.read_csv(DATA/'records.csv').fillna({'sampling_province':'Unknown'})
    i = pd.read_csv(DATA/'items.csv')
    # The event-level network uses one normalized item per report, not repeated aliases.
    d = i.drop_duplicates(['record_id','item_en']).merge(r.drop(columns=['year','product_en']), on='record_id', validate='many_to_one')
    return r, i, d

def bn_data(province=False):
    r,i,d = load_data()
    b = pd.DataFrame({'HazardCat':d.category_en,'Item':d.item_en,
      'ProdR':d.production_province.map(PROVINCES).fillna('Unknown') if province else d.production_region,
      'SampR':d.sampling_province.map(PROVINCES).fillna('Unknown') if province else d.sampling_region,
      'Product':d.product_en,
      'Stage':d.sampling_stage.map({'流通环节':'Distribution','生产环节':'Production','餐饮环节':'Catering'}).fillna('Unknown'),
      'group_id':d.group_id,'record_id':d.record_id,'year':d.year,
      'Client':d.sampling_province.map(PROVINCES).fillna('Unresolved location').replace('Unknown','Unresolved location')})
    return b.fillna('Unknown')
