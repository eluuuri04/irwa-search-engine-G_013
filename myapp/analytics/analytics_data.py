import json
import time
import uuid
import altair as alt
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional


class AnalyticsData:
    """
    In-memory persistence object for analytics.
    Dimensions (dim_*): master data tables
    Facts (fact_*): event logs
    """

    def __init__(self):
        # Dimensions
        self.dim_queries: Dict[str, Dict] = {}
        self.dim_documents: Dict[str, Dict] = {}
        self.dim_contexts: Dict[str, Dict] = {}

        # Facts
        self.fact_requests: List[Dict] = []
        self.fact_queries: List[Dict] = []
        self.fact_result_impressions: List[Dict] = []
        self.fact_clicks_rows: List[Dict] = []
        self.fact_dwells: List[Dict] = []

        # Legacy + derived
        self.fact_clicks: Dict[str, int] = {}
        self._last_click_ts: Dict[tuple, float] = {}

        # Internal clicks
        self.fact_internal_clicks: defaultdict = defaultdict(int)
        self.fact_internal_click_rows: List[Dict] = []

    def _now(self) -> float:
        return time.time()

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def save_query_terms(self, terms: str) -> str:
        query_id = self._new_id()
        tokens = [t for t in terms.split() if t]
        self.dim_queries[query_id] = {
            'query_id': query_id,
            'terms': tokens,
            'term_count': len(tokens),
            'order': list(range(len(tokens))),
        }
        self.fact_queries.append({'ts': self._now(), 'query_id': query_id})
        return query_id

    def ensure_document(self, doc_id: str, title: str = '', url: str = ''):
        if doc_id and doc_id not in self.dim_documents:
            self.dim_documents[doc_id] = {'doc_id': doc_id, 'title': title or '', 'url': url or ''}

    def ensure_context(self, browser: Optional[str], os_name: Optional[str],
                       device: Optional[str], time_bucket: Optional[str],
                       ip: Optional[str] = None, country: Optional[str] = None,
                       city: Optional[str] = None) -> str:
        b = browser or 'unknown'
        o = os_name or 'unknown'
        d = device or 'unknown'
        t = time_bucket or 'unknown'
        i = ip or 'unknown'
        ctx_id = '|'.join([b, o, d, t, i])
        if ctx_id not in self.dim_contexts:
            self.dim_contexts[ctx_id] = {
                'context_id': ctx_id,
                'browser': b,
                'os': o,
                'device': d,
                'time_bucket': t,
                'ip': i,
                'country': country,
                'city': city,
            }
        return ctx_id

    def log_request(self, path: str, method: str, status: int, context_id: str):
        self.fact_requests.append({
            'ts': self._now(), 'path': path, 'method': method, 'status': status, 'context_id': context_id
        })

    def log_result_impressions(self, query_id: str, ranked_docs: List[Dict]):
        ts = self._now()
        for rank, doc in enumerate(ranked_docs or [], start=1):
            doc_id = doc.get('pid')
            if not doc_id:
                continue
            self.ensure_document(doc_id, doc.get('title', ''), doc.get('url', ''))
            self.fact_result_impressions.append({
                'ts': ts, 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
            })

    def log_click(self, query_id: str, doc_id: str, rank: Optional[int] = None):
        ts = self._now()
        if not doc_id:
            return
        self.fact_clicks_rows.append({
            'ts': ts, 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
        })
        self.fact_clicks[doc_id] = self.fact_clicks.get(doc_id, 0) + 1
        
        # NOMÉS guardar el primer timestamp si no existeix
        key = (query_id, doc_id)
        if key not in self._last_click_ts:
            self._last_click_ts[key] = ts

    def log_return_to_results(self, query_id: str, doc_id: str):
        key = (query_id, doc_id)
        ts = self._last_click_ts.get(key)
        if ts is not None:
            dwell = max(0.0, self._now() - ts)
            self.fact_dwells.append({
                'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'dwell_seconds': dwell
            })
            del self._last_click_ts[key]
    def log_internal_click(self, doc_id: str, element: Optional[str] = None, meta: Optional[dict] = None):
        if not doc_id:
            return
        self.fact_internal_clicks[doc_id] += 1
        safe_meta = meta if isinstance(meta, dict) else {}
        self.fact_internal_click_rows.append({
            'ts': self._now(), 'doc_id': doc_id, 'element': element, 'meta': safe_meta
        })


    # ---------- Charts ----------
    def chart_query_length_distribution(self):
        df = pd.DataFrame([{'term_count': q['term_count']} for q in self.dim_queries.values()])
        if df.empty:
            df = pd.DataFrame({'term_count': []})
        return alt.Chart(df).mark_area(opacity=0.6).encode(
            x='term_count:Q', y='count():Q'
        ).properties(title='Distribution of query term counts')

    def chart_ctr_by_rank(self):
        imp = pd.DataFrame(self.fact_result_impressions)
        clk = pd.DataFrame(self.fact_clicks_rows)
        if imp.empty:
            return alt.Chart(pd.DataFrame({'rank': [], 'ctr': []})).mark_line()
        imp_g = imp.groupby('rank').size().rename('impressions')
        clk_g = clk.groupby('rank').size().rename('clicks') if not clk.empty else pd.Series(dtype='int64', name='clicks')
        df = pd.concat([imp_g, clk_g], axis=1).fillna(0).reset_index()
        df['ctr'] = (df['clicks'] / df['impressions']).fillna(0)
        return alt.Chart(df).mark_line(point=True).encode(x='rank:Q', y='ctr:Q').properties(title='CTR by rank')

    def chart_dwell_distribution(self):
        df = pd.DataFrame(self.fact_dwells)
        if df.empty or 'dwell_seconds' not in df.columns:
            df = pd.DataFrame({'dwell_seconds': []})
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('dwell_seconds:Q', bin=alt.Bin(maxbins=30)), y='count():Q'
        ).properties(title='Dwell time distribution (seconds)')

    def chart_number_of_views(self):
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame({'Document ID': [], 'Number of Views': []})
        return alt.Chart(df).mark_bar().encode(
            x='Document ID:N', y='Number of Views:Q'
        ).properties(title='Number of Views per Document')

    def chart_internal_clicks_by_doc(self):
        data = [{'Document ID': doc_id, 'Internal Clicks': count} for doc_id, count in self.fact_internal_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame({'Document ID': [], 'Internal Clicks': []})
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y'), y='Internal Clicks:Q'
        ).properties(title='Internal Clicks per Document')

    def dashboard_html(self):
        charts = alt.vconcat(
            self.chart_query_length_distribution(),
            self.chart_ctr_by_rank(),
            self.chart_dwell_distribution(),
            self.chart_number_of_views(),
            self.chart_internal_clicks_by_doc()
        )
        return charts.to_html()


class ClickedDoc:
    def __init__(self, doc_id: str, description: str, views: int = 0, internal_clicks: int = 0):
        self.doc_id = doc_id
        self.description = description
        self.views = views
        self.internal_clicks = internal_clicks

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__)
