import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGGenerator:

    PROMPT_TEMPLATE = """
You are an expert product advisor for an e-commerce website.

### Your Objective:
Recommend the single best product that satisfies the user's request **as precisely as possible**.

### Mandatory Hard Rules:
- MUST match product type (t-shirt must not be shorts, shoes, pants, etc.)
- MUST match user-intended gender (men vs women)
- MUST match color if included in the user query
- Prefer matches for material (cotton blend, polyester, etc.)

### Selection Priorities (after hard rules):
1️⃣ Closest match to user intent (color, material, fit, style)
2️⃣ Quality indicators (rating, brand)
3️⃣ Price fairness and value for money

### Disqualification Rules:
Do NOT select products that:
- Do not match category/type
- Are for the wrong gender
- Are a worse attribute match when a better one exists

If **no** product is a valid match, respond EXACTLY:
"There are no good products that fit the request based on the retrieved results."

### Retrieved Products:
{retrieved_results}

### User Request:
{user_query}

### Output Format (strict):
- Best Product: [PID] [Product Name]
- Why: [Precise attribute-based justification]
- Alternative: [Optional: only if strongly relevant]
"""


    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> str:

        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."

        try:
            if not retrieved_results:
                return "There are no good products that fit the request based on the retrieved results."

            query_lower = user_query.lower()
            filtered = []

            # ---------------------------
            # 🔥 HARD FILTER: PRODUCT TYPE + GENDER
            # ---------------------------
            for res in retrieved_results:
                title = getattr(res, "title", "").lower()
                desc = getattr(res, "description", "").lower()
                text = title + " " + desc

                # Must be a T-Shirt
                if not ("t shirt" in text or "t-shirt" in text or "tshirt" in text):
                    continue

                # Gender filtering
                if "men" in query_lower and "men" not in text:
                    continue
                if "women" in query_lower and "women" not in text:
                    continue

                filtered.append(res)

            # ---------------------------
            # 🔥 HARD FILTER: COLOR
            # ---------------------------
            color_keywords = ["grey", "gray", "blue", "black", "white", "red", "yellow", "green", "pink"]
            detected_colors = [c for c in color_keywords if c in query_lower]

            if detected_colors:
                color_filtered = []
                for res in filtered:
                    text = getattr(res, "title", "").lower() + getattr(res, "description", "").lower()
                    if any(color in text for color in detected_colors):
                        color_filtered.append(res)

                # Only override if we found exact color matches
                if color_filtered:
                    filtered = color_filtered

            # Fallback — if filters removed everything
            final_results = filtered[:top_N] if filtered else retrieved_results[:top_N]

            # ---------------------------
            # Format results for LLM
            # ---------------------------
            formatted_results = "\n".join([
                f"- PID: {res.pid} | "
                f"Name: {getattr(res, 'title', 'N/A')} | "
                f"Price: {getattr(res, 'selling_price', 'N/A')} | "
                f"Rating: {getattr(res, 'average_rating', 'N/A')} | "
                f"Brand: {getattr(res, 'brand', 'N/A')} | "
                f"Category: {getattr(res, 'category', 'N/A')} | "
                f"Info: {(getattr(res, 'description', '')[:120] + '...') if hasattr(res, 'description') else 'N/A'}"
                for res in final_results
            ])

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results,
                user_query=user_query,
            )

            # ---------------------------
            # Groq API Client Call
            # ---------------------------
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER
