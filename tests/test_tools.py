from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_empty_wardrobe

def test_search_returns_results():
    results = search_listings("vintage", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0

def test_search_empty_results():
    # Deliberate failure mode: impossible criteria
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []   # Should return empty list without crashing

def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=45.0)
    assert all(item["price"] <= 45.0 for item in results)

def test_suggest_outfit_empty_wardrobe():
    # Test failure mode: agent shouldn't crash if wardrobe is empty
    mock_item = {"title": "Test Shirt", "category": "tops"}
    empty_wardrobe = get_empty_wardrobe()
    
    result = suggest_outfit(mock_item, empty_wardrobe)
    assert isinstance(result, str)
    assert len(result) > 10 # Ensures the LLM actually said something

def test_create_fit_card_empty_outfit():
    # Test failure mode: guard clause should catch empty strings
    mock_item = {"title": "Test Shirt", "price": 10, "platform": "depop"}
    result = create_fit_card("", mock_item)
    
    assert "Could not generate fit card" in result