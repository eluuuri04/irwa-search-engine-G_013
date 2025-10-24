# **README — Text Processing and EDA for Fashion Products Dataset**

## **Project Overview**

This project processes and searches through a fashion e-commerce dataset (JSON), applying text preprocessing techniques (lowercasing, cleaning, tokenization, stopwords removal, lemmatization, and stemming). It also builds an inverted index and performs exploratory data analysis (EDA) with visualizations. Additionally, it evaluates the coverage of search results against a validation file (validation\_labels.csv).

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
├─ Project_part1_group13.ipynb
├─ fashion_products_dataset.json # dataset 
├─ validation_labels.csv # CSV with ground-truth validation
├─ README.md
```

The main notebook includes all the functional code. To execute it successfully, ensure that the dataset (fashion\_products\_dataset.json) and the validation file (validation\_labels.csv) are available in your Google Drive workspace.

**Quick Start: Running on Google Colab** 

1. Mount Google Drive in Colab (already done in our notebook in cell \[1\]):

```py
from google.colab import drive
drive.mount('/content/drive')
```

2. Adjust dataset and validation paths:

```py
docs_path = '/content/drive/MyDrive/fashion_products_dataset.json'
validation_path = '/content/drive/MyDrive/validation_labels.csv'
```
These data paths can be changed, depending on where the notebook is executed. The rest of the code will execute normally, independently on the paths. 

3. Run the cells with imports and nltk.download() (cell \[2\]).

4. Build the index by running the cell that defines build\_terms() and create\_index(), then:

```py
index, products_info = create_index(lines)
```

5. Run the search and evaluation cells (search(), queries, and validation evaluation with validation\_labels.csv): see cells \[8\], \[9\], \[12\] in our notebook.

**Main Functions**

- ### **build\_terms(text)**

Text preprocessing applied to title \+ description. Modifiable options within the function: Enable/disable lemmatization, enable/disable stemming, stopwords list, remove numbers or short tokens.

- ### **create\_index(lines)**

Builds the inverted index and the products\_info dictionary. Returns (index, products\_info).

- ### **search(query, index, products\_info)**

Preprocesses the query and returns a list of product IDs that appear in the postings of any of the terms.

**Evaluation with validation\_labels.csv**

Our notebook already implements logic to calculate coverage (the percentage of relevant product IDs from the validation file that appear in your results for each query).

```
# load validation_labels.csv -> validation_data
# construct sets: set_label1_rel, set_label2_rel
pids_q1 = set(r['pid'] for r in results_q1)
coverage_q1 = len(set_label1_rel & pids_q1)/len(set_label1_rel)*100
```

This prints coverage and the list of found/missing product IDs.
