"""Audit notification chronology without reclassifying or removing failures.

The recent window is fixed relative to the extraction cutoff, not the run date.
Shares use noncompliant reports; no item-specific testing denominator is assumed.
"""
from bee_common import *

CUTOFF = pd.Timestamp('2026-09-20')
RECENT_START = pd.Timestamp('2021-09-21')
PERIODS = ['2015-01-01 to 2021-09-20', '2021-09-21 to 2026-09-20']

def main():
    r, i, d = load_data()
    for frame in (r, d):
        frame['notification_date'] = pd.to_datetime(frame['date'], errors='raise')
        assert frame.notification_date.le(CUTOFF).all()
        frame['period'] = np.where(frame.notification_date.lt(RECENT_START), PERIODS[0], PERIODS[1])
    totals = r.groupby('period').size()
    rows = []
    for period in PERIODS:
        sub = d[d.period.eq(period)]
        for category in CATS:
            n = sub.loc[sub.category_en.eq(category), 'record_id'].nunique()
            rows.append({'period': period, 'category': category, 'reports_with_category': n,
                         'all_noncompliant_reports_in_period': int(totals[period]),
                         'share_of_noncompliant_reports': n / totals[period]})
    save(pd.DataFrame(rows), 'period_category_audit')
    item = d.groupby(['period', 'item_en', 'category_en']).agg(
        reports_with_item=('record_id', 'nunique'), first_notification=('notification_date', 'min'),
        last_notification=('notification_date', 'max')).reset_index()
    item['all_noncompliant_reports_in_period'] = item.period.map(totals)
    item['share_of_noncompliant_reports'] = item.reports_with_item / item.all_noncompliant_reports_in_period
    save(item, 'period_item_audit')
    lead = d[d.item_en.eq('Lead')].copy().sort_values('notification_date')
    lead = lead.rename(columns={'source_row_x':'source_workbook_row','source_link_x':'announcement_link'})
    lead = lead[['record_id','source_workbook_row','notification_date','year','product_en','period',
                 'value_raw','limit_raw','concentration_ugkg','quantitative_primary','announcement_link']]
    lead['date_basis'] = 'Notification date recorded in the platform export'
    save(lead, 'lead_chronology_audit')
    meta = {'extraction_cutoff': str(CUTOFF.date()), 'recent_window_start': str(RECENT_START.date()),
            'lead_reports_total': len(lead), 'lead_reports_earlier': int(lead.period.eq(PERIODS[0]).sum()),
            'lead_reports_recent': int(lead.period.eq(PERIODS[1]).sum()),
            'lead_reports_recent_dates': lead.loc[lead.period.eq(PERIODS[1]),'notification_date'].dt.strftime('%Y-%m-%d').tolist(),
            'earlier_window_noncompliant_reports': int(totals[PERIODS[0]]),
            'recent_window_noncompliant_reports': int(totals[PERIODS[1]]),
            'interpretation': 'Descriptive notification chronology; not an analyte prevalence trend or a policy-effect estimate.'}
    jsave(meta, 'period_audit_protocol')
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
