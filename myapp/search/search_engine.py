import random
import numpy as np
from myapp.search.algorithms import build_terms, create_index_with_tfidf, rank_products_custom2
from myapp.search.objects import Document


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res


class SearchEngine:
    """Class that implements the REAL search engine logic"""

    def search(self, search_query, search_id, corpus):
        global INDEX, PRODUCTS_INFO, TF, DF, IDF
        
        print("Search query:", search_query)
        if 'INDEX' not in globals() or INDEX is None:
            print("Creant l'índex per primer cop...")
            corpus_dicts = [
                {
                    "pid": doc.pid,
                    "title": doc.title,
                    "description": doc.description,
                    "brand": getattr(doc, "brand", ""),
                    "category": getattr(doc, "category", ""),
                    "sub_category": getattr(doc, "sub_category", ""),
                    "product_details": getattr(doc, "product_details", ""),
                    "seller": getattr(doc, "seller", ""),
                    "out_of_stock": getattr(doc, "out_of_stock", False),
                    "selling_price": str(getattr(doc, "selling_price") or 0),
                    "actual_price": str(getattr(doc, "actual_price") or 0),
                    "discount": str(getattr(doc, "discount") or 0),
                    "average_rating": getattr(doc, "average_rating", None),
                    "url": doc.url
                }
                for doc in corpus.values()
            ]
            INDEX, PRODUCTS_INFO, TF, DF, IDF = create_index_with_tfidf(corpus_dicts)
        else:
            print("Índex ja creat, usant-lo directament...")
            
        # 1. Preprocess query
        query_terms = build_terms(search_query)

        if not query_terms:
            return []

        # 2. Intersect documents
        first_term = query_terms[0]
        if first_term not in INDEX:
            return []

        docs = set(posting[0] for posting in INDEX[first_term])

        for term in query_terms[1:]:
            if term not in INDEX:
                return []
            term_docs = set(posting[0] for posting in INDEX[term])
            docs &= term_docs

        if not docs:
            return []

        # 3. Rank using CUSTOM2
        ranked_docs, scores = rank_products_custom2(
            query_terms,
            INDEX,
            IDF,
            TF,
            PRODUCTS_INFO,
            list(docs)
        )

        # 4. Convert to Document objects for Flask
        results = []
        for pid in ranked_docs[:20]:
            prod = PRODUCTS_INFO[pid]

            results.append(
                Document(
                    pid=pid,
                    title=prod["title"],
                    description=prod["description"],
                    url="doc_details?pid={}&search_id={}".format(pid, search_id),
                    ranking=1.0
                )
            )

        return results

###############################################################################


# class SearchEngine:
#     """Class that implements the search engine logic"""

#     def search(self, search_query, search_id, corpus):
#         print("Search query:", search_query)

#         results = []
#         ### You should implement your search logic here:
#         #results = dummy_search(corpus, search_id)  # replace with call to search algorithm

#         # results = search_in_corpus(search_query)
#         return results
