import os
from json import JSONEncoder
import httpagentparser
from flask import Flask, render_template, session, request
from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()  # take environment variables from .env

# --- JSON encoder patch for to_json methods ---
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

def _ensure_context_and_log_request(path: str, method: str, status: int):
    user_agent = request.headers.get('User-Agent')
    agent = httpagentparser.detect(user_agent) or {}
    browser = agent.get('browser', {}).get('name', 'unknown')
    os_name = agent.get('os', {}).get('name', 'unknown')
    device = 'mobile' if agent.get('platform') == 'mobile' else 'desktop'
    ctx_id = analytics_data.ensure_context(browser, os_name, device, _time_bucket())
    analytics_data.log_request(path, method, status, ctx_id)

# --- Routes ---
@app.route('/')
def index():
    session['some_var'] = "Some value that is kept in session"
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)
    print("Remote IP:", user_ip, "Agent:", agent)
    _ensure_context_and_log_request(path='/', method=request.method, status=200)
    return render_template('index.html', page_title="Welcome")

@app.route('/search', methods=['POST'])
def search_form_post():
    _ensure_context_and_log_request(path='/search', method=request.method, status=200)
    search_query = request.form['search-query']
    session['last_search_query'] = search_query
    query_id = analytics_data.save_query_terms(search_query)
    results = search_engine.search(search_query, query_id, corpus)
    analytics_data.log_result_impressions(query_id, [
        {'pid': doc.pid, 'title': doc.title, 'url': doc.url} for doc in results
    ])
    rag_response = rag_generator.generate_response(search_query, results)
    found_count = len(results)
    session['last_found_count'] = found_count
    return render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_response=rag_response,
        query_id=query_id
    )

@app.route('/doc_details', methods=['GET'])
def doc_details():
    _ensure_context_and_log_request(path='/doc_details', method=request.method, status=200)
    clicked_doc_id = request.args["pid"]
    query_id = request.args.get("qid")
    rank_str = request.args.get("rank", "0")
    try:
        rank = int(rank_str)
    except:
        rank = 0
    print(f"Click: doc={clicked_doc_id}, query={query_id}, rank={rank}")
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 0
    if query_id:
        analytics_data.log_click(query_id=query_id, doc_id=clicked_doc_id, rank=rank)
    return render_template('doc_details.html')

@app.route('/return_to_results', methods=['GET'])
def return_to_results():
    _ensure_context_and_log_request(path='/return_to_results', method=request.method, status=200)
    query_id = request.args.get("qid")
    doc_id = request.args.get("pid")
    if query_id and doc_id:
        analytics_data.log_return_to_results(query_id=query_id, doc_id=doc_id)
    return ('', 204)

@app.route('/stats', methods=['GET'])
def stats():
    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title,
                            description=row.description, url=row.url, count=count)
        docs.append(doc)
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)
    dashboard_html = analytics_data.dashboard_html()  # Altair charts
    return render_template('dashboard.html',
                           visited_docs=visited_docs,
                           dashboard_html=dashboard_html,
                           page_title="Dashboard")

@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()

@app.route('/analytics', methods=['GET'])
def analytics_dashboard():
    _ensure_context_and_log_request(path='/analytics', method=request.method, status=200)
    html = analytics_data.dashboard_html()
    return render_template('dashboard.html',
                           page_title="Analytics",
                           dashboard_html=html)

# --- Run ---
if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
