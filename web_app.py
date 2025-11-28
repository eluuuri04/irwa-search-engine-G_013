import os
import requests
from json import JSONEncoder
import httpagentparser
from flask import Flask, render_template, session, request, jsonify, redirect, url_for
from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --- JSON encoder patch ---
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default

# --- Flask app setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")

# --- Core components ---
search_engine = SearchEngine()
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()

# --- Load corpus ---
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])

# --- Helpers ---
def _time_bucket():
    return f"{datetime.utcnow().hour:02d}:00"

def _get_location_from_ip(ip: str):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json")
        data = response.json()
        return data.get("country", "unknown"), data.get("city", "unknown")
    except:
        return "unknown", "unknown"

def _ensure_context_and_log_request(path: str, method: str, status: int):
    user_agent = request.headers.get('User-Agent')
    agent = httpagentparser.detect(user_agent) or {}
    browser = agent.get('browser', {}).get('name', 'unknown')
    os_name = agent.get('os', {}).get('name', 'unknown')
    device = 'mobile' if agent.get('platform') == 'mobile' else 'desktop'
    ip = request.remote_addr
    country, city = _get_location_from_ip(ip)
    ctx_id = analytics_data.ensure_context(browser, os_name, device, _time_bucket(),
                                           ip=ip, country=country, city=city)
    analytics_data.log_request(path, method, status, ctx_id)

# --- Routes ---
@app.route('/')
def index():
    session['some_var'] = "Some value that is kept in session"
    _ensure_context_and_log_request(path='/', method=request.method, status=200)
    return render_template('index.html', page_title="Welcome")

# --- SEARCH POST ---
@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form.get('search-query', '').strip()
    session['last_search_query'] = search_query
    # No fem analytics aquí → PRG
    return redirect(url_for('search_results_get', search_query=search_query))

# --- SEARCH GET ---
@app.route('/search_results', methods=['GET'])
def search_results_get():
    raw_query = request.args.get('search_query', '').strip()

    if not raw_query:
        return render_template("results.html",
                               results_list=[],
                               found_counter=0,
                               rag_response=None,
                               query_id=None,
                               raw_query="",
                               corrected_query="")

    # ✨ 1️⃣ Correct the query using AI
    corrected_query = rag_generator.normalize_query(raw_query)

    # Save both versions in session
    session['raw_query'] = raw_query
    session['corrected_query'] = corrected_query

    # 2️⃣ Analytics logs store the corrected query (higher quality data)
    try:
        query_id = analytics_data.save_query_terms(corrected_query)
    except:
        query_id = None

    # 3️⃣ Retrieve with corrected query
    results_list, results = [], []
    try:
        results = search_engine.search(corrected_query, query_id, corpus)
    except Exception:
        results = []

    # Convert results for UI
    for res in results:
        doc = corpus.get(res.pid)
        if doc:
            results_list.append({
                'pid': doc.pid,
                'title': doc.title,
                'description': getattr(doc, 'description', ''),
                'url': getattr(doc, 'url', ''),
                'selling_price': getattr(doc, 'selling_price', 'N/A'),
                'discount': getattr(doc, 'discount', ''),
                'average_rating': getattr(doc, 'average_rating', 'N/A'),
                'seller': getattr(doc, 'seller', '—'),
                'images': getattr(doc, 'images', [])
            })

    # 4️⃣ Log impressions
    if query_id:
        try:
            analytics_data.log_result_impressions(query_id, [
                {'pid': d['pid'], 'title': d['title'], 'url': d['url']}
                for d in results_list
            ])
        except Exception:
            pass

    # 5️⃣ RAG ranking using corrected query
    try:
        rag_response = rag_generator.generate_response(corrected_query, results)
    except Exception:
        rag_response = None

    return render_template("results.html",
                           results_list=results_list,
                           found_counter=len(results_list),
                           rag_response=rag_response,
                           query_id=query_id,
                           raw_query=raw_query,
                           corrected_query=corrected_query)


# --- DOC DETAILS ---
@app.route('/doc_details', methods=['GET'])
def doc_details():
    _ensure_context_and_log_request(path='/doc_details', method=request.method, status=200)

    clicked_doc_id = request.args.get("pid")
    query_id = request.args.get("qid")
    rank_str = request.args.get("rank", "0")

    try:
        rank = int(rank_str)
    except:
        rank = 0

    # Log click (start dwell)
    if query_id and clicked_doc_id:
        analytics_data.log_click(query_id=query_id, doc_id=clicked_doc_id, rank=rank)

    doc = corpus.get(clicked_doc_id)
    if not doc:
        return render_template(
            'doc_details.html', 
            doc=None, 
            query_id=query_id, 
            page_title="Document not found"
        ), 404

    return render_template(
        'doc_details.html',
        doc=doc,
        query_id=query_id,     # ✅ ESSENTIAL for dwell tracking
        page_title=getattr(doc, 'title', 'Document')
    )

# --- LOG INTERNAL CLICK ---
@app.route('/log_internal_click', methods=['GET','POST'])
def log_internal_click():
    pid, element, meta = None, None, None
    if request.method=="GET":
        pid = request.args.get("pid")
        element = request.args.get("element")
    else:
        try:
            data = request.get_json(force=True)
            pid = data.get("pid")
            element = data.get("element")
            meta = data.get("meta")
        except:
            pid = request.form.get("pid")
            element = request.form.get("element")

    if not pid:
        return jsonify({"error":"pid required"}),400

    analytics_data.log_internal_click(doc_id=pid, element=element, meta=meta)
    return ('',204)

# --- RETURN TO RESULTS ---
@app.route('/return_to_results', methods=['GET'])
def return_to_results():
    print("🔥 /return_to_results HIT:", request.args)   # DEBUG LOG

    _ensure_context_and_log_request(path='/return_to_results', method=request.method, status=200)

    query_id = request.args.get("qid")
    doc_id   = request.args.get("pid")

    if query_id and doc_id:
        analytics_data.log_return_to_results(query_id=query_id, doc_id=doc_id)

    return ('', 204)

@app.route("/log_dwell_time", methods=["POST"])
def log_dwell_time():
    pid = request.form.get("pid")
    query_id = request.form.get("query_id")
    dwell_seconds = request.form.get("dwell_seconds")

    if not pid or not dwell_seconds:
        return jsonify({"error": "pid and dwell_seconds required"}), 400

    try:
        dwell_seconds = float(dwell_seconds)
    except ValueError:
        dwell_seconds = 0.0

    # Append to fact_dwells directly
    analytics_data.fact_dwells.append({
        "ts": analytics_data._now(),
        "query_id": query_id,
        "doc_id": pid,
        "dwell_seconds": dwell_seconds
    })

    return "", 204


# --- STATS ---
@app.route('/stats', methods=['GET'])
def stats():
    docs = []
    for doc_id, count in analytics_data.fact_clicks.items():
        row: Document = corpus.get(doc_id)
        if row:
            doc = StatsDocument(pid=row.pid, title=row.title,
                                description=row.description, url=row.url, count=count)
        else:
            doc = StatsDocument(pid=doc_id, title=doc_id, description="", url="", count=count)
        docs.append(doc)
    docs.sort(key=lambda doc: doc.count, reverse=True)

    queries = []
    for q in analytics_data.dim_queries.values():
        queries.append({
            "query_id": q["query_id"],
            "terms": " ".join(q["terms"]),
            "term_count": q["term_count"]
        })

    return render_template('stats.html', clicks_data=docs, queries_data=queries, page_title="Quick Stats")

# --- DASHBOARD ---
@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    all_doc_ids = set(list(analytics_data.fact_clicks.keys()) + list(analytics_data.fact_internal_clicks.keys()))
    for doc_id in all_doc_ids:
        d: Document = corpus.get(doc_id)
        views = analytics_data.fact_clicks.get(doc_id, 0)
        internal = analytics_data.fact_internal_clicks.get(doc_id, 0)
        desc = getattr(d,'description','') if d else ''
        doc = ClickedDoc(doc_id, desc, views, internal)
        visited_docs.append(doc)

    visited_docs.sort(key=lambda doc: (doc.internal_clicks, doc.views), reverse=True)
    dashboard_html = analytics_data.dashboard_html()
    return render_template('dashboard.html',
                           visited_docs=visited_docs,
                           dashboard_html=dashboard_html,
                           page_title="Dashboard")

# --- PLOT VIEWS ---
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.chart_number_of_views().to_html()

# --- ANALYTICS DASHBOARD ---
@app.route('/analytics', methods=['GET'])
def analytics_dashboard():
    _ensure_context_and_log_request(path='/analytics', method=request.method, status=200)
    html = analytics_data.dashboard_html()
    return render_template('dashboard.html', page_title="Analytics", dashboard_html=html)

# --- METADATA ---
@app.route('/metadata/<pid>', methods=['GET'])
def metadata(pid):
    _ensure_context_and_log_request(path='/metadata', method=request.method, status=200)
    doc = corpus.get(pid)
    if not doc:
        return jsonify({"error": "not found"}), 404
    try:
        if hasattr(doc, "to_json"):
            return jsonify(doc.to_json())
        if isinstance(doc, dict):
            return jsonify(doc)
        return jsonify(doc.__dict__)
    except Exception as e:
        return jsonify({"error": "could not serialize", "detail": str(e)}), 500

# --- RESULTS TEMPLATE ---
@app.route("/results")
def results():
    return render_template("results.html")

# --- RUN ---
if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))