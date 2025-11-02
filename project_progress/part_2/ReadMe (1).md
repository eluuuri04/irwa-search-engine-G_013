# **README — Text Processing, Ranking & Evaluation**

## **Project Overview**

This second part of the project extends the search system developed in Part 1, adding new functionalities:

* Calculation of TF-IDF weights and document vectorization.  
* Cosine similarity–based search to rank results by relevance.  
* Implementation of an alternative ranking model using BM25 (Best Matching 25\)  
* Performance evaluation using metrics (*Precision, Recall, F1-score, MAP*).  
* Efficiency and performance comparison with the search from Part 1\.  
* Comparison between TF-IDF and BM25 in terms of returned documents  
* Use of a second validation file (our\_queries\_validation.csv) containing our own queries for testing.

The project is based on the same dataset (fashion\_products\_dataset.json).

## **Requirements**

Install the following libraries (Colab already has many of them pre-installed):

```
pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn
```

In Google Colab, you just need to run nltk.download(...) as shown in our notebook.  
In our notebook the lines for installing and importing are already ready to be executed.

## **Project Structure**

```
project-root/
│
├─ Project_part2_group13.ipynb
├─ fashion_products_dataset.json
├─ validation_labels.csv
├─ our_queries_validation.csv
├─ README.md
```

The main notebook contains all the functional code.  
 Make sure the dataset and validation files are accessible from your working environment (Google Drive or local).

## **Quick Start: Running on Google Colab**

1. Mount Google Drive:

```
from google.colab import drive
drive.mount('/content/drive')
```

2. Set dataset and validation paths:

```
 docs_path = '/content/drive/MyDrive/fashion_products_dataset.json'
validation_path = '/content/drive/MyDrive/validation_labels.csv'
our_validation_path = '/content/drive/MyDrive/our_queries_validation.csv'
```

3. Run the import and preprocessing cells (NLTK, libraries, etc.).

4. Build the TF-IDF matrix and inverted index:

```
index, products_info, tf, df, idf = create_index_with_tfidf(lines)
```

5. Search and rank results:

```
ranked_docs, doc_scores = search_products_tf_idf(query, index, idf, tf, products_info)

# Using BM25 ranking
ranked_docs, doc_scores = search_products(query, index, idf, tf, products_info, ranking_method="bm25")

```

   

6. Evaluate results using the validation files

## **Main Functions**

- ### **build\_terms(text)**

Text preprocessing applied to title \+ description. Modifiable options within the function: Enable/disable lemmatization, enable/disable stemming, stopwords list, remove numbers or short tokens.

- **create\_index\_with\_tfidf(lines)**

Builds an inverted index and calculates TF-IDF weights for all terms in each document. It outputs several structures: the term index with document positions, a products\_info dictionary containing metadata, term frequency (TF) and document frequency (DF) tables, and inverse document frequency (IDF) values. This function prepares all data needed for cosine similarity ranking.

- **rank\_products(query\_terms, docs, index, idf, tf, products\_info)**

Computes cosine similarity between the query vector and document vectors. It creates a TF-IDF vector for the query, constructs weighted document vectors, calculates similarity scores, and sorts products in descending order of relevance, returning ranked product IDs with their scores.

- **search\_products\_tf\_idf(query, index, idf, tf, products\_info)**

Implements the complete TF-IDF search pipeline. The function preprocesses the query, identifies documents containing all query terms, calls the ranking function to compute cosine similarity, and returns a list of product IDs ranked by relevance.

- **search\_products(query, index, idf, tf, products\_info, ranking\_method="bm25")**

Extends the search functionality by allowing a choice between BM25 and TF-IDF ranking methods. The function preprocesses the user query, retrieves documents containing all query terms, and ranks them using the selected similarity model. When ranking\_method is set to "bm25", it applies the BM25 ranking formula; otherwise, it defaults to TF-IDF cosine similarity. The output includes a list of product IDs sorted by relevance and their corresponding similarity scores.

- **Evaluation Functions**

Includes functions to compute key information retrieval metrics such as Precision, Recall, F1-score, Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), and Coverage. These metrics are used to assess search performance using both validation files.

## **Evaluation & Output**

The notebook outputs:

* Ranked search results by relevance.

* Coverage and performance metrics compared against both validation\_labels.csv and our\_queries\_validation.csv.  
