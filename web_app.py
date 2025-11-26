import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Helper for analytics context
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


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)

    _ensure_context_and_log_request(path='/', method=request.method, status=200)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    _ensure_context_and_log_request(path='/search', method=request.method, status=200)

    search_query = request.form['search-query']
    session['last_search_query'] = search_query

    # create query_id and store query terms
    query_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(search_query, query_id, corpus)

    # Log impressions (ranked by position in results)
    analytics_data.log_result_impressions(query_id, [
        {'pid': doc.pid, 'title': doc.title, 'url': doc.url} for doc in results
    ])

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

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
    print("doc details session: ")
    print(session)

    res = session["some_var"]
    print("recovered var from session:", res)

    clicked_doc_id = request.args["pid"]
    query_id = request.args.get("qid")
    rank_str = request.args.get("rank", "0")
    try:
        rank = int(rank_str)
    except:
        rank = 0

    print(f"click in id={clicked_doc_id}, query_id={query_id}, rank={rank}")

    # store data in statistics table 1
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    # structured click event
    if query_id:
        analytics_data.log_click(query_id=query_id, doc_id=clicked_doc_id, rank=rank)

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
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
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
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

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


@app.route('/analytics', methods=['GET'])
def analytics_dashboard():
    _ensure_context_and_log_request(path='/analytics', method=request.method, status=200)
    html = analytics_data.dashboard_html()
    return render_template('dashboard.html', page_title="Analytics", dashboard_html=html)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
