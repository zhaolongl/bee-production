"""Product-specific odds ratios of non-compliance from grouped-binomial regression.

Observed product-month counts of non-compliant and compliant batches are modelled
directly; no individual compliant rows are generated. Honey is the reference,
notification year is a categorical covariate and standard errors are clustered
by notification month. The estimates are associations among reported inspections.
"""
import patsy
import statsmodels.api as sm
from bee_common import *

FORMULA = "C(product_en, Treatment(reference='Honey')) + C(year)"

def fit(df):
    x=patsy.dmatrix(FORMULA,df,return_type='dataframe')
    model=sm.GLM(df[['failed','passed']].to_numpy(),x,family=sm.families.Binomial())
    return model.fit(cov_type='cluster',cov_kwds={'groups':df.date})

def main():
    df=pd.read_csv(DATA/'product_monthly.csv')
    df=df[(df.year>=2015)&(df.total>0)].copy()
    rows=[]
    for label,subset in [('Complete years 2015-2025',df[df.year<=2025]),('Including partial 2026',df)]:
        result=fit(subset);ci=result.conf_int()
        for term in result.params.index:
            if 'product_en' not in term:continue
            rows.append(dict(analysis=label,product_en=term.split('[T.')[1][:-1],reference='Honey',
                  odds_ratio=float(np.exp(result.params[term])),ci_low=float(np.exp(ci.loc[term,0])),
                  ci_high=float(np.exp(ci.loc[term,1])),p_value=float(result.pvalues[term]),
                  adjustment='Categorical notification year',covariance='Clustered by notification month',
                  n_cells=len(subset),n_month_clusters=subset.date.nunique(),n_total=int(subset.total.sum())))
    print(save(pd.DataFrame(rows),'product_associations').to_string(index=False))

if __name__=='__main__':main()
