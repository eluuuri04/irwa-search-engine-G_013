import json
import time
import uuid
import altair as alt
import pandas as pd
from collections import defaultdict


class AnalyticsData:
    """
    In-memory persistence object for analytics.
    """

    # Keep your original dict for compatibility (views when opening doc_detail)
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

        # New: clicks that happen *inside* a document (internal clicks)
        self.fact_internal_clicks = defaultdict(int)  # doc_id -> internal click count
        self.fact_internal_click_rows = []  # optional: {ts, doc_id, element, meta}

        # Derived
        self.doc_click_counts = defaultdict(int)
        self._last_click_ts = {}

    def _now(self):
        return time.time()

    def _new_id(self):
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
            self.ensure_document(doc_id, doc.get('title', ''), doc.get('url', ''))
            self.fact_result_impressions.append({
                'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
            })

    def log_click(self, query_id: str, doc_id: str, rank: int):
        """
        Logged when a user clicks and opens the document details (a view).
        Keeps backward compatibility with fact_clicks dict and fact_clicks_rows.
        """
        self.fact_clicks_rows.append({
            'ts': self._now(), 'query_id': query_id, 'doc_id': doc_id, 'rank': rank
        })
        # increment derived counters (but keep previous behaviour of setting to 0 if used elsewhere)
        self.doc_click_counts[doc_id] += 1
        # maintain legacy fact_clicks dict (if other code expects it); increment
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

    # ---------- New: internal click logging ----------
    def log_internal_click(self, doc_id: str, element: str = None, meta: dict = None):
        """
        Register a click that happens *inside* the document view (e.g., click on images,
        expand buttons, links inside the doc). This is independent from opening the doc.
        element: optional string describing the clicked element/id.
        meta: optional dictionary with other info.
        """
        self.fact_internal_clicks[doc_id] += 1
        self.fact_internal_click_rows.append({
            'ts': self._now(), 'doc_id': doc_id, 'element': element, 'meta': meta or {}
        })

    # ---------- Charts ----------

    def chart_query_length_distribution(self):
        df = pd.DataFrame([{'term_count': q['term_count']} for q in self.dim_queries.values()])
        if df.empty:
            df = pd.DataFrame({'term_count': []})
        return alt.Chart(df).mark_area(opacity=0.6).encode(
            x='term_count:Q',
            y='count():Q'
        ).properties(title='Distribution of query term counts')

    def chart_ctr_by_rank(self):
        imp = pd.DataFrame(self.fact_result_impressions)
        clk = pd.DataFrame(self.fact_clicks_rows)
        if imp.empty:
            return alt.Chart(pd.DataFrame({'rank': [], 'ctr': []})).mark_line()
        imp_g = imp.groupby('rank').size().rename('impressions')
        clk_g = clk.groupby('rank').size().rename('clicks')
        df = pd.concat([imp_g, clk_g], axis=1).fillna(0).reset_index()
        df['ctr'] = (df['clicks'] / df['impressions']).fillna(0)
        return alt.Chart(df).mark_line(point=True).encode(
            x='rank:Q', y='ctr:Q'
        ).properties(title='CTR by rank')

    def chart_dwell_distribution(self):
        df = pd.DataFrame(self.fact_dwells)
        if df.empty:
            df = pd.DataFrame({'dwell_seconds': []})
        return alt.Chart(df).mark_bar().encode(
            x=alt.X('dwell_seconds:Q', bin=alt.Bin(maxbins=30)),
            y='count():Q'
        ).properties(title='Dwell time distribution (seconds)')

    def plot_number_of_views(self):
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty: df = pd.DataFrame({'Document ID': [], 'Number of Views': []})
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID:N',
            y='Number of Views:Q'
        ).properties(
            title='Number of Views per Document'
        )
        return chart.to_html()

    def chart_internal_clicks_by_doc(self):
        """
        Altair chart showing internal clicks per document.
        """
        data = [{'Document ID': doc_id, 'Internal Clicks': count} for doc_id, count in self.fact_internal_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame({'Document ID': [], 'Internal Clicks': []})
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y'),
            y=alt.Y('Internal Clicks:Q')
        ).properties(title='Internal Clicks per Document')
        return chart

    def dashboard_html(self):
        """
        Build a vertical concatenation of Altair charts including internal clicks.
        """
        charts = alt.vconcat(
            self.chart_query_length_distribution(),
            self.chart_ctr_by_rank(),
            self.chart_dwell_distribution(),
            self.chart_internal_clicks_by_doc()
        )
        return charts.to_html()

    # NOTE: there's another ensure_context below in original file; keep compatibility by redefining:
    def ensure_context(self, browser: str, os_name: str, device: str, time_bucket: str,
                       ip: str = None, country: str = None, city: str = None):
        key = (browser or 'unknown', os_name or 'unknown', device or 'unknown',
               time_bucket or 'unknown', ip or 'unknown')
        ctx_id = '|'.join(key)
        if ctx_id not in self.dim_contexts:
            self.dim_contexts[ctx_id] = {
                'context_id': ctx_id,
                'browser': browser,
                'os': os_name,
                'device': device,
                'time_bucket': time_bucket,
                'ip': ip,
                'country': country,
                'city': city,
            }
        return ctx_id


class ClickedDoc:
    """
    Data structure passed to the dashboard template.
    Contains both the number of times the doc was opened (views) and the number of internal clicks.
    """

    def __init__(self, doc_id, description, views: int = 0, internal_clicks: int = 0):
        self.doc_id = doc_id
        self.description = description
        self.views = views
        self.internal_clicks = internal_clicks

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__)
