

````
### README — Advanced Ranking, Word2Vec & Evaluation

### Project Overview

This third part of the project extends the search and ranking system developed in Parts 1 and 2. We keep the TF‑IDF index and basic search pipeline from Part 2, and add several new ranking strategies (TF‑IDF, BM25, two customized variants) as well as a semantic ranking approach based on Word2Vec. We also expand the evaluation framework to compare all methods (TF‑IDF, BM25, Custom1, Custom2, Word2Vec) using multiple relevance definitions provided in `our_queries_validation.csv`. The goal is to understand how different ranking models behave on the fashion product dataset `fashion_products_dataset.json`, both in terms of exact lexical matching and more semantic similarity.

### Requirements

You need the following Python libraries (most are preinstalled in Colab):

```bash
pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn gensim
````

In Google Colab, you also have to download the NLTK resources used by the preprocessing functions:

```py
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
```

All install/import and NLTK download cells are already prepared in the notebook.

### **Project Structure**

```
project-root/
│
├─ Project_part3_group13.ipynb
├─ fashion_products_dataset.json
├─ our_queries_validation.csv
├─ validation_labels.csv
├─ README.md
```

The notebook `Project_part3_group13.ipynb` contains the complete code for index creation, ranking methods (TF‑IDF, BM25, Custom1, Custom2, Word2Vec), and evaluation. The JSON dataset and validation CSVs must be accessible from your environment (e.g. mounted Google Drive paths).

### **Quick Start: Running on Google Colab**

1. **Mount Google Drive** so the dataset and validation files can be read:

```py
from google.colab import drive
drive.mount('/content/drive')
```

2.   
   **Set the paths** to the main dataset and validation files:

```py
docs_path = '/content/drive/MyDrive/fashion_products_dataset.json'
validation_path = '/content/drive/MyDrive/validation_labels.csv'
our_validation_path = '/content/drive/MyDrive/our_queries_validation.csv'
```

3.   
   **Run the setup cells** for imports, installing `gensim`, and downloading the NLTK data. Then load the JSON dataset into a `lines` variable using `json.load`.

4. **Build the TF‑IDF index** and related structures by calling:

```py
index, products_info, tf, df, idf = create_index_with_tfidf(lines)
```

5.   
   **Run text‑based search** over a small set of test queries with different ranking methods, for example:

```py
queries = {
    1: "men cotton blend grey t shirt",
    2: "light blue jeans for men slim fit",
    3: "women casual wear cotton shirt",
    4: "printed navy top for women",
    5: "regular fit denim jeans"
}

methods = ["tf-idf", "bm25", "custom1", "custom2"]

for qid, query in queries.items():
    for method in methods:
        ranked_docs, scores = search_products(
            query, index, idf, tf, products_info, ranking_method=method
        )
        # inspect top results in the notebook output
```

6.   
   **Train the Word2Vec model and run semantic ranking** by first building a preprocessed corpus, training the model, converting each product to a dense vector, and finally ranking with cosine similarity:

```py
corpus_tokens = []
for doc in lines:
    text = f"{doc.get('title', '')} {doc.get('description', '')}"
    corpus_tokens.append(build_terms(text))

model = train_custom_word2vec(corpus_tokens)

doc_vectors = []
for doc in lines:
    pid = doc.get("pid", "")
    text = f"{doc.get('title', '')} {doc.get('description', '')}"
    terms = build_terms(text)
    vec = tokens_to_vector(terms, model)
    doc_vectors.append((pid, vec))

rankings = word2vec_ranking_topk(
    model=model,
    doc_vectors=doc_vectors,
    products_info=products_info,
    queries=queries,
    top_k=20
)
```

7.   
   **Run the evaluation cells** to compute Precision@k, Recall@k, and F1@k for all ranking methods and relevance strategies. The notebook will output detailed per‑query tables and averaged metrics.

### **Main Functions**

#### **Text Preprocessing**

The function `build_terms(text)` is responsible for cleaning and normalizing the text (title \+ description) of each product. It converts the text to lowercase, strips punctuation and special characters, normalizes whitespace, splits the text into tokens, removes English stopwords using NLTK, and then applies both lemmatization and stemming. The output is a list of processed tokens that are used both for the classical (TF‑IDF, BM25, custom) rankings and as input to train the Word2Vec model.

#### **Index Construction and TF‑IDF**

The function `create_index_with_tfidf(lines)` traverses all documents in the dataset and builds several key data structures. It creates an inverted index from each term to a list of documents and term positions, and fills a `products_info` dictionary mapping each product ID (`pid`) to its metadata (title, description, brand, category, prices, discount, rating, URL, etc.). It also computes a term frequency table `tf[term][pid]` with log‑scaled TF values, a document frequency table `df[term]`, and the corresponding inverse document frequencies `idf[term] = log(N / df[term])`. These structures are the basis for TF‑IDF cosine ranking and are reused by the other text‑based methods.

### **Ranking Approaches (Text‑Based)**

**TF‑IDF ranking (`rank_products_tf_idf`)**  
 This function builds a TF‑IDF weighted vector for the query and for each candidate document, using the inverted index and the `tf` and `idf` tables. It then computes cosine similarity between the query vector and each document vector and returns products sorted by similarity score.

**BM25 ranking (`rank_products_bm25`)**  
 This method computes a BM25 score for each candidate document based on term frequency, document length, average document length and IDF. It accumulates contributions for all query terms and orders products by their final BM25 score, often improving ranking quality over pure TF‑IDF for longer texts.

**Custom ranking 1 (`rank_products_custom1`)**  
 Custom1 first runs BM25 to get a base text relevance score for each product, then adjusts it using product metadata such as normalized rating and discount. By combining BM25 with these attributes through a weighted formula, it promotes well‑rated and more discounted products while still respecting textual relevance.

**Custom ranking 2 (`rank_products_custom2`)**  
 Custom2 starts from a TF‑IDF–based score but adds domain‑specific boosts and penalties using metadata like title matches, average rating, review count, price, discount, and stock status. The final score multiplies textual relevance by these factors so that highly rated, reasonably priced, in‑stock products with strong title matches are ranked higher.

### **Unified Search Function**

The function `search_products(query, index, idf, tf, products_info, ranking_method)` provides a single entry point to all text‑based ranking methods. It first preprocesses the user query with `build_terms` and uses the inverted index to find the set of candidate documents containing all query terms. Depending on the value of `ranking_method` (“tf-idf”, “bm25”, “custom1”, or “custom2”), it then delegates to the corresponding ranking function described above and returns both the ordered list of product IDs and the associated scores. This makes it easy to switch between ranking models while keeping the query processing and candidate selection steps the same.

### **Semantic Ranking with Word2Vec**

`train_custom_word2vec(corpus_tokens, vector_size=50)` trains a Word2Vec model on the preprocessed product texts, where `corpus_tokens` is a list of token lists (one per document). The function configures hyperparameters such as window size, minimum frequency, and number of workers, and returns the word vectors `model.wv`. Once the model is trained, `tokens_to_vector(tokens, model)` converts any sequence of tokens (either from a product or a query) into a single dense vector by averaging the Word2Vec embeddings of all tokens present in the vocabulary, returning a zero vector when no token is known.

On top of these building blocks, `word2vec_ranking_topk(model, doc_vectors, products_info, queries, top_k=20)` performs the actual semantic ranking. It assumes that `doc_vectors` already contains a pair `(pid, vector)` for each product, where `vector` is the averaged Word2Vec representation of its title and description. For each query in the `queries` dictionary, the function preprocesses the text with `build_terms`, computes its vector with `tokens_to_vector`, and then calculates cosine similarity between the query vector and every document vector. The documents are then sorted by similarity, and the top‑k results (including their product metadata and scores) are returned for each query, enabling retrieval of semantically related products even when there is no exact keyword match.

### **Evaluation Functions and Relevance Strategies**

The evaluation code uses `our_queries_validation.csv`, which lists for each `query_id` a set of candidate product IDs (`pid`) and several relevance labels. The functions `precision_at_k(doc_score, y_score, k=10)`, `recall_at_k(doc_score, y_score, k=10)` and `f1_at_k(doc_score, y_score, k=10)` take a binary ground‑truth vector `doc_score` (1 for relevant, 0 for non‑relevant) and a prediction score vector `y_score` aligned with the same product IDs, and compute Precision@k, Recall@k and F1@k by considering only the top‑k documents ranked by `y_score`. For the four text‑based methods, `y_score` is obtained from the ranking order (higher score for higher rank, with small random noise to break ties), and for Word2Vec, it is derived from the cosine similarity ranking.

The file `our_queries_validation.csv` provides multiple “relevance strategies” through different columns, such as “Exact coincidence”, “All \- 1”, “\>= 70%”, and “product types”, which reflect increasingly relaxed definitions of what counts as relevant. The evaluation loops over all queries and all ranking methods (TF‑IDF, BM25, Custom1, Custom2, and Word2Vec), and for each combination it computes Precision@k, Recall@k, and F1@k under each relevance strategy. The results are stored in `results_df` for the text‑based methods and `w2v_df` for Word2Vec, and then averaged per method and strategy, producing tables (`avg_results`, `w2v_avg`, and `overall_avg`) that summarize how each ranking approach performs depending on how strictly relevance is defined.

### **Evaluation & Output**

When the notebook is fully executed, it outputs several types of results. For each query and ranking method, it prints the top ranked products together with their titles, scores, and key metadata (such as rating, discount, or price), which allows a qualitative inspection of how the different models behave (for example, whether Custom2 really pushes highly rated, well‑priced items forward). In addition, the evaluation section generates pandas DataFrames with per‑query metrics (Precision@10, Recall@10, F1@10) for every method and relevance strategy, as well as aggregated tables that show average performance across all queries. By comparing these tables, we can see, for instance, that some methods may do better when strict exact matching is required, while others (especially Word2Vec) can provide stronger recall and F1 under looser, more semantic relevance definitions, giving a more complete view of the trade‑offs between lexical and semantic ranking approaches in our fashion product search engine.

```

```

