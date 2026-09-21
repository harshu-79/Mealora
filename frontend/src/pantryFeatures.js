import { supabase } from "./supabaseClient";

function getDaysToExpiry(expiryDate) {
  if (!expiryDate) {
    return null;
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const expiry = new Date(`${expiryDate}T00:00:00`);
  expiry.setHours(0, 0, 0, 0);

  const difference =
    expiry.getTime() - today.getTime();

  return Math.ceil(
    difference / (1000 * 60 * 60 * 24)
  );
}

function calculateExpiryUrgency(daysToExpiry) {
  if (daysToExpiry === null) {
    return 0;
  }

  if (daysToExpiry <= 0) {
    return 1;
  }

  if (daysToExpiry <= 3) {
    return 0.9;
  }

  if (daysToExpiry <= 7) {
    return 0.6;
  }

  if (daysToExpiry <= 14) {
    return 0.3;
  }

  return 0.1;
}

export async function getPantryFeatures(userId = null) {
  let query = supabase
    .from("pantry_items")
    .select("*");

  if (userId) {
    query = query.eq("user_id", userId);
  }

  const { data, error } = await query;

  if (error) {
    console.error(
      "Error loading pantry features:",
      error
    );

    throw error;
  }

  const items = data || [];

  const enrichedItems = items.map((item) => {
    const daysToExpiry =
      getDaysToExpiry(item.expiry_date);

    return {
      ...item,

      days_to_expiry: daysToExpiry,

      expiry_urgency:
        calculateExpiryUrgency(
          daysToExpiry
        ),

      is_expired:
        daysToExpiry !== null &&
        daysToExpiry <= 0,

      is_expiring_soon:
        daysToExpiry !== null &&
        daysToExpiry <= 7,

      is_low_stock:
        item.low_stock_threshold !== null &&
        item.low_stock_threshold !== undefined &&
        Number(item.quantity) <=
          Number(item.low_stock_threshold),
    };
  });

  const expiringItems =
    enrichedItems.filter(
      (item) =>
        item.is_expiring_soon
    );

  const lowStockItems =
    enrichedItems.filter(
      (item) =>
        item.is_low_stock
    );

  return {
    ingredients:
      enrichedItems,

    normalized_ingredients:
      enrichedItems.map(
        (item) =>
          item.normalized_name
      ),

    total_items:
      enrichedItems.length,

    total_categories:
      new Set(
        enrichedItems
          .map(
            (item) =>
              item.category
          )
          .filter(Boolean)
      ).size,

    expiring_items:
      expiringItems,

    low_stock_items:
      lowStockItems,

    total_expiry_urgency:
      enrichedItems.reduce(
        (sum, item) =>
          sum +
          item.expiry_urgency,
        0
      ),
  };
}