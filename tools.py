"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""
from __future__ import annotations
import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    # 1. Load the data
    listings = load_listings()
    results = []
    
    # 2. Extract keywords from the user's description (lowercased)
    keywords = description.lower().split()

    for item in listings:
        # Filter by price
        if max_price is not None and item["price"] > max_price:
            continue
            
        # Filter by size (case-insensitive substring match)
        if size is not None:
            item_size = item.get("size", "").lower()
            if size.lower() not in item_size:
                continue

        # 3. Score by keyword overlap
        score = 0
        search_text = (
            item["title"].lower() + " " + 
            item["description"].lower() + " " + 
            " ".join(item.get("style_tags", []))
        )
        
        for word in keywords:
            if word in search_text:
                score += 1
                
        # 4. Keep if there's any match
        if score > 0:
            item["_score"] = score # temporarily store score for sorting
            results.append(item)

    # 5. Sort by score descending and remove the temporary score key
    results.sort(key=lambda x: x["_score"], reverse=True)
    for res in results:
        del res["_score"]

    return results


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    # Replace this with your implementation
    client = _get_groq_client()
    
    item_details = f"{new_item['title']} ({new_item['category']})"
    
    # 1. Handle the Empty Wardrobe Failure Mode gracefully
    if not wardrobe.get("items"):
        system_prompt = (
            "You are an expert fashion stylist. The user just bought a new item but has an empty digital wardrobe. "
            "Give them 2-3 general styling tips on what kinds of clothes pair well with this item."
        )
        user_prompt = f"The item is: {item_details}. How should I style this?"
    else:
        # 2. Handle the Happy Path
        wardrobe_str = "\n".join([f"- {w['name']} ({w['category']})" for w in wardrobe["items"]])
        system_prompt = (
            "You are an expert fashion stylist. Suggest 1-2 complete outfits using the user's NEW item "
            "and pieces specifically chosen from their EXISTING wardrobe."
        )
        user_prompt = f"NEW ITEM: {item_details}\n\nEXISTING WARDROBE:\n{wardrobe_str}"

    # 3. Call the LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    # Replace this with your implementation
# 1. Guard against empty string (Failure Mode handling)
    if not outfit or not outfit.strip():
        return "Could not generate fit card: outfit details are missing."
        
    client = _get_groq_client()
    
    system_prompt = (
        "You are a Gen-Z fashion influencer writing a casual, authentic OOTD caption. "
        "Mention the thrifted item, its price, and the platform. Capture the vibe of the outfit. "
        "Keep it to 2-4 sentences. Use lowercase for an aesthetic vibe. Use 1-2 emojis max."
    )
    
    user_prompt = (
        f"Item: {new_item['title']} (${new_item['price']} from {new_item['platform']}).\n"
        f"Outfit Vibe: {outfit}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9  # High temperature so it sounds different every time!
    )
    
    return response.choices[0].message.content.strip()
