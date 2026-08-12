// Curated quick-pick items, grouped by category. Every keyword here was
// checked against the real catalog (data_real/fashion_products_kcore3.csv)
// to confirm it substring-matches at least one product title, so these
// aren't just plausible-sounding labels -- they're verified to resolve to
// a real item, not silently fall through to the semantic-match/popularity
// fallback path. Short, clean labels (not full scraped titles) also match
// more reliably against recommend_for_history()'s substring search than
// copying a full noisy Amazon title would.
window.CATALOG_SUGGESTIONS = [
  {
    category: "Winter Wear",
    items: ["Winter Coat", "Beanie", "Puffer Jacket", "Fleece Jacket", "Gloves", "Boots"],
  },
  {
    category: "Athletic",
    items: ["Running Shoes", "Yoga Pants", "Sports Bra", "Leggings", "Joggers", "Tank Top"],
  },
  {
    category: "Formalwear",
    items: ["Dress Shirt", "Blazer", "Oxford Shoes", "Button Down", "Cardigan", "Polo Shirt"],
  },
  {
    category: "Swimwear & Beach",
    items: ["Swim Trunk", "Bikini", "Flip Flop", "Swimsuit", "Sandals"],
  },
  {
    category: "Accessories",
    items: ["Sunglasses", "Leather Belt", "Wrist Watch", "Necklace", "Earrings", "Bracelet", "Backpack", "Baseball Cap"],
  },
  {
    category: "Casual",
    items: ["T-Shirt", "Jeans", "Sneakers", "Hoodie", "Sweatshirt", "Denim Jacket", "Maxi Dress"],
  },
  {
    category: "Loungewear",
    items: ["Pajama", "Robe", "Bodysuit", "Socks"],
  },
  {
    category: "Kids",
    items: ["Kids", "Toddler", "Girls Boots", "Boys Sunglasses"],
  },
];
