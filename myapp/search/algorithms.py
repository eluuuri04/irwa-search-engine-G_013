def build_terms(line):
    """
    Preprocess the text (title + description):
    - Lowercase
    - Remove punctuation and special symbols (e.g., ₹, %, ™)
    - Normalize spaces
    - Tokenization
    - Remove stop words
    - Lemmatization
    - Stemming
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase
    line = line.lower()

    # Remove punctuation and uncommon symbols
    line = re.sub(r"[^\w\s]", " ", line)

    # Normalize spaces
    line = re.sub(r"\s+", " ", line).strip()

    # Tokenize
    tokens = line.split()

    # Remove stop words
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatization (reduces words to their base form)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # Stemming (we don't need, explained in the report)
    tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def create_index_with_tfidf(lines):
    # Initialize data structures
    index = defaultdict(list)
    products_info = {}
    tf = defaultdict(dict)
    df = defaultdict(int)
    idf = defaultdict(float)

    num_documents = len(lines)

    for doc in lines:
        pid = doc.get("pid", "")
        title = doc.get("title", "")
        description = doc.get("description", "")
        brand = doc.get("brand", "")
        category = doc.get("category", "")
        sub_category = doc.get("sub_category", "")
        product_details = doc.get("product_details", "")
        seller = doc.get("seller", "")
        out_of_stock = doc.get("out_of_stock", "")
        selling_price_str = doc.get("selling_price", "").replace(',', '')
        selling_price = float(selling_price_str) if selling_price_str else 0.0
        discount_str = doc.get("discount", "").strip('% off')
        discount = float(discount_str) if discount_str else 0.0
        actual_price_str = doc.get("actual_price", "").replace(',', '')
        actual_price = float(actual_price_str) if actual_price_str else 0.0
        average_rating_str = doc.get("average_rating", "")
        average_rating = float(average_rating_str) if average_rating_str else None
        url = doc.get("url", "")

        text = f"{title} {description}"
        terms = build_terms(text)

        products_info[pid] = {
            "pid": pid,
            "title": title,
            "description": description,
            "brand": brand,
            "category": category,
            "sub_category": sub_category,
            "product_details": product_details,
            "seller": seller,
            "out_of_stock": out_of_stock,
            "selling_price": selling_price,
            "discount": discount,
            "actual_price": actual_price,
            "average_rating": average_rating,
            "url": url
        }

        # Compute term frequency and positional index for the current document
        term_freq = defaultdict(int)
        current_page_index = {}

        for position, term in enumerate(terms):
            term_freq[term] += 1
            try:
                current_page_index[term][1].append(position)
            except:
                current_page_index[term] = [pid, array('I', [position])]

        # Update inverted index and document frequency for each term
        for term, posting in current_page_index.items():
            df[term] += 1
            index[term].append(posting)

        # Compute TF for each term in the doc
        for term, freq in term_freq.items():
            if freq > 0:
                tfidf_weight = (1 + math.log(freq))  # IDF will be applied later
                tf[term][pid] = round(tfidf_weight, 4)
            else:
                tf[term][pid] = 0.0

    # Compute IDF for each term
    for term in df:
        idf[term] = round(math.log(num_documents / df[term]), 4)

    return index, products_info, tf, df, idf


def rank_products_custom2(query_terms, index, idf, tf, products_info, docs):
    doc_scores = defaultdict(float)

    for pid in docs:
        info = products_info[pid]

        title_text = info["title"].lower()
        description_text = info["description"].lower()

        #base textual score (TF-IDF)
        base_score = 0.0

        for term in query_terms:
            if term in tf and pid in tf[term]:
                tfidf_score = tf[term][pid] * idf.get(term, 0.0)

                #Title weighting applied
                if term in title_text:
                    tfidf_score *= 2.0

                base_score += tfidf_score

        #rating + review
        rating = float(info.get("average_rating", 0) or 0.0)
        rating_norm = rating / 5.0

        review_count = float(info.get("rating_count", 1.0) or 1.0)
        review_boost = math.log1p(review_count) / 3.0

        social_proof = 1 + (0.4 * rating_norm) * (0.3 + review_boost)

        #price and discount
        price = float(info.get("selling_price", 1.0) or 1.0)
        if price <= 0:
            price = 1.0

        price_penalty = 1 / (1 + math.log1p(price))

        discount = float(info.get("discount", 0.0) or 0.0) / 100.0
        discount_boost = 1 + 0.2 * discount


        #if no stock we don't want the product
        stock_boost = 0 if info.get("out_of_stock", False) else 1.0

        #final score
        final_score = (base_score * social_proof * discount_boost * price_penalty * stock_boost)

        doc_scores[pid] = final_score

    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    result_docs = [pid for pid, _ in ranked_docs]

    return result_docs, ranked_docs
