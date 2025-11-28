import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert product advisor for an e-commerce website.

### Your Objective:
Recommend the single best product that satisfies the user's request as accurately as possible.

### Important Attribute Usage:
When deciding the best product:
- HIGH priority to matching: gender, color, material, product category/type
- USE additional metadata to justify selection: price/value, rating, brand
- Prefer higher rated or better value product when attributes are similar

### Mandatory Safety Rule:
If NONE of the retrieved products meet the key user intent (gender, product type, and color when provided):
respond EXACTLY:

"There are no good products that fit the request based on the retrieved results."

### Corrected Query:
{corrected_query}

### Retrieved Products:
{retrieved_results}

### Output Format (strict):
- Best Product: [PID] [Product Name]
- Why: [Precise justification referencing only visible attributes: color, style, rating, price, brand, etc.]
- Alternative: [Optional only if strongly relevant and must match intent]
"""

    def normalize_query(self, query: str) -> str:
        """Correct spelling and grammar using Groq."""
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            prompt = f"""
Correct the spelling and grammar of this shopping query:

"{query}"

Return ONLY the corrected text, no explanation.
"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return query

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> str:
        corrected_query = self.normalize_query(user_query)
        query_lower = corrected_query.lower()

        # Early exit if nothing available
        if not retrieved_results:
            return "There are no good products that fit the request based on the retrieved results."

        # Hard Filtering using gender + color (required intent)
        gender_terms = ["men", "women", "male", "female"]
        detected_gender = [g for g in gender_terms if g in query_lower]

        color_terms = ["grey", "gray", "blue", "black", "white", "red", "yellow", "green", "pink"]
        detected_colors = [c for c in color_terms if c in query_lower]

        filtered = []
        for res in retrieved_results:
            text = f"{res.title} {res.description}".lower()

            if detected_gender:
                if not any(g in text for g in detected_gender):
                    continue

            if detected_colors:
                if not any(c in text for c in detected_colors):
                    continue

            filtered.append(res)

        # Fallback: If filter eliminates everything → no valid product
        if not filtered:
            return "There are no good products that fit the request based on the retrieved results."

        final_results = filtered[:top_N]

        # Include all metadata available to improve ranking accuracy
        formatted_docs = "\n".join([
            f"- PID: {r.pid} | Name: {r.title} | Price: {getattr(r, 'selling_price', 'N/A')} | "
            f"Rating: {getattr(r, 'average_rating', 'N/A')} | Brand: {getattr(r, 'brand', 'N/A')} | "
            f"Category: {getattr(r, 'category', 'N/A')} | Color: {getattr(getattr(r, 'product_details', {}), 'Color', 'N/A')}"
            for r in final_results
        ])

        prompt = self.PROMPT_TEMPLATE.format(
            corrected_query=corrected_query,
            retrieved_results=formatted_docs,
        )

        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except Exception:
            return "AI Ranking unavailable — please try again."
