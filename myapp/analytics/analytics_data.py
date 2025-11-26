import json
import random
import time
import uuid
import altair as alt
import pandas as pd
from collections import defaultdict


class AnalyticsData:
    """
    In-memory persistence object for analytics.
    """

    # Keep your original dict for compatibility
    fact_clicks = dict([])

    def __init__(self):
        # Dimensions
        self.dim_queries = {}        # query_id -> {terms, term_count, order}
        self.dim_documents = {}      # doc_id -> {title, url}
        self.dim_contexts = {}       # context_id -> {browser, os, device, time_bucket}

        # Facts
        self.fact_requests = []           # {ts, path, method, status, context_id}
        self.fact_queries = []            # {ts, query_id}
        self.fact_result_impressions = [] # {ts, query_id, doc_id, rank}
        self.fact_clicks_rows = []        # {ts, query_id, doc_id, rank}
        self.fact_dwells = []             # {ts, query_id, doc_id, dwell_seconds}

        # Derived
        self.doc_click_counts = defaultdict(int)
        self._last_click_ts = {}

    def _now(self): return time.time()
    def _new_id(self): return str(uuid.uuid4())

    def save_query_terms(self, terms: str) -> str:
        """
        Save query terms and return a query_id.
        """
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
        if doc_id not in self.dim_documents:
            self.dim_documents[doc_id] = {'doc_id': doc_id, 'title': title, 'url': url}

    def ensure_context(self, browser: str, os_name: str, device: str, time_bucket: str):
        key = (browser or 'unknown', os_name or 'unknown', device or 'unknown', time_bucket or 'unknown')
        ctx_id = '|'.join(key)
        if ctx_id not in self.dim_contexts:
            self.dim_contexts[ctx_id] = {
                'context_id': ctx_id,
                'browser': key[0],
                'os': key[1],
                'device': key[2],
                'time_bucket': key[3],
            }
        return ctx_id

    def log_request(self, path: str, method: str, status: int, context_id: str):
        self.fact_requests.append({
            'ts': self._now(), 'path': path, 'method': method, 'status': status, 'context_id': context_id
        })

    def log_result_impressions(self, query_id: str, ranked_docs: list):
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_id = doc['pid']
            self.ensure_document(doc_id, doc.get('title',''), doc.get('url',''))
            self.fact_result_impressions.append({
                'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
            })

    def log_click(self, query_id: str, doc_id: str, rank: int):
        self.fact_clicks_rows.append({
            'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
        })
        self.doc_click_counts[doc_id] += 1
        # Maintain compatibility with old dict
        if doc_id in self.fact_clicks.keys():
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1
        self._last_click_ts[(query_id, doc_id)] = self._now()

    def log_return_to_results(self, query_id: str, doc_id: str):
        key = (query_id, doc_id)
        ts = self._last_click_ts.get(key)
        if ts is not None:
            dwell = max(0, self._now() - ts)
            self.fact_dwells.append({
                'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'dwell_seconds': dwell
            })
            del self._last_click_ts[key]

    # ---------- Charts ----------
    def chart_doc_clicks(self):
        df = pd.DataFrame([{'Document ID': d, 'Clicks': c} for d, c in self.doc_click_counts.items()])
        if df.empty: df = pd.DataFrame({'Document ID':[], 'Clicks':[]})
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y'),
            y='Clicks:Q'
        ).properties(title='Clicks per document')

    def chart_query_length_distribution(self):
        df = pd.DataFrame([{'term_count': q['term_count']} for q in self.dim_queries.values()])
        if df.empty: df = pd.DataFrame({'term_count':[]})
        return alt.Chart(df).mark_area(opacity=0.6).encode(
            x='term_count:Q',
            y='count():Q'
        ).properties(title='Distribution of query term counts')

    def chart_ctr_by_rank(self):
        imp = pd.DataFrame(self.fact_result_impressions)
        clk = pd.DataFrame(self.fact_clicks_rows)
        if imp.empty:
            return alt.Chart(pd.DataFrame({'rank':[], 'ctr':[]})).mark_line()
        imp_g = imp.groupby('rank').size().rename('impressions')
        clk_g = clk.groupby('rank').size().rename('clicks')
        df = pd.concat([imp_g, clk_g], axis=1).fillna(0).reset_index()
        df['ctr'] = (df['clicks'] / df['impressions']).fillna(0)
        return alt.Chart(df).mark_line(point=True).encode(
            x='rank:Q', y='ctr:Q'
        ).properties(title='CTR by rank')

    def chart_dwell_distribution(self):
        df = pd.DataFrame(self.fact_dwells)
        if df.empty: df = pd.DataFrame({'dwell_seconds':[]})
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('dwell_seconds:Q', bin=alt.Bin(maxbins=30)),
            y='count():Q'
        ).properties(title='Dwell time distribution (seconds)')

    def plot_number_of_views(self):
        # Keep your original method
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        return chart.to_html()

    def dashboard_html(self):
        charts = alt.vconcat(
            self.chart_doc_clicks(),
            self.chart_query_length_distribution(),
            self.chart_ctr_by_rank(),
            self.chart_dwell_distribution()
        )
        return charts.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self)
