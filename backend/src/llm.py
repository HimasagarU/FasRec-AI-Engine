import os
import json
from groq import Groq
from fastapi import HTTPException

# We will initialize the client lazily
client = None

def generate_outfit_narration(query_product: dict, recommended_products: list[dict],
                               available_article_types: list[str] = None) -> dict:
    """
    Calls Groq API (LLaMA 3.3-70B) to generate a structured JSON outfit recommendation.
    Returns structured outfit pieces using REAL article type names from our catalog
    so the backend can search for matching products.
    """
    global client
    if not client:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
        client = Groq(api_key=api_key)

    # Format the query product details
    q_name = query_product.get("title", "Unknown Product")
    q_category = query_product.get("masterCategory", "")
    q_sub = query_product.get("subCategory", "")
    q_article = query_product.get("articleType", "")
    q_color = query_product.get("baseColour", "")
    q_gender = query_product.get("gender", "")
    q_usage = query_product.get("usage", "")
    q_season = query_product.get("season", "")

    # Format the top 5 recommended products
    top_5 = recommended_products[:5]
    recs_json = json.dumps([
        {
            "title": p.get("title"),
            "articleType": p.get("articleType"),
            "color": p.get("baseColour"),
        } for p in top_5
    ], indent=2)

    # Build the available types constraint
    types_constraint = ""
    if available_article_types:
        # Send a curated subset to keep prompt reasonable
        sample_types = available_article_types[:80]
        types_constraint = f"""
CRITICAL RULE — You MUST pick the "type" field for each outfit piece from this EXACT list of article types available in our catalog. Do NOT invent or modify type names:
{json.dumps(sample_types)}

Pick the closest matching type from the list above. For example, use "Jeans" not "Dark Brown Jeans", use "Casual Shoes" not "Casual Boots", use "Belts" not "Brown Leather Belt", use "Watches" not "Minimalist Watch".
"""

    prompt = f"""You are an expert personal fashion stylist.

The user is viewing this product:
- Name: {q_name}
- Category: {q_category} > {q_sub} > {q_article}
- Color: {q_color}
- Gender: {q_gender}
- Usage: {q_usage}
- Season: {q_season}

Similar items from our catalog (for context):
{recs_json}
{types_constraint}
Your task: Create a COMPLETE outfit around the query product appropriate for {q_gender}, {q_usage} usage, and {q_season} season.

Suggest 4-6 complementary pieces. Do NOT re-suggest "{q_article}" (the query product type).

For each piece, specify:
- "type": MUST be an exact value from the available article types list above
- "color": a specific color that pairs well (e.g., "Navy Blue", "Black", "Brown", "White")
- "why": one short sentence explaining the pairing

Also write:
- "recommendation": A 2-3 sentence stylist recommendation tying the whole outfit together
- "occasion": The ideal occasion (short phrase)

Respond in strict JSON:
{{
  "recommendation": "...",
  "occasion": "...",
  "outfit_pieces": [
    {{"type": "...", "color": "...", "why": "..."}},
    ...
  ]
}}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert fashion stylist AI. Always respond in valid JSON. Use ONLY the exact article type names provided — never invent new ones."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        
        response_content = chat_completion.choices[0].message.content
        result = json.loads(response_content)
        return result
    except Exception as e:
        print(f"[LLM Error] {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate outfit narration: {str(e)}")
