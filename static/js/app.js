// ══════════════════════════════════════════════════════════
//  Smart Food Waste System — Main UI Script
// ══════════════════════════════════════════════════════════

// Set default date to today
document.getElementById('date').valueAsDate = new Date();

// ── Tab switching ─────────────────────────────────────────

function switchTab(tab) {
    document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

    document.getElementById('tab-' + tab).classList.add('active');
    document.getElementById('tab-' + tab + '-btn').classList.add('active');

    // Scroll to top smoothly
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Occupancy slider ──────────────────────────────────────

document.getElementById('occupancy').addEventListener('input', function () {
    document.getElementById('occupancy-value').textContent = this.value;
});

// ── Facility type labels ──────────────────────────────────

document.querySelectorAll('input[name="facility_type"]').forEach(radio => {
    radio.addEventListener('change', function () {
        document.getElementById('facilityTypeLabel').textContent =
            this.value === 'hostel' ? 'Hostel' : 'Restaurant';
    });
});

// ══════════════════════════════════════════════════════════
//  TAB 1 — MEAL DEMAND PREDICTION
// ══════════════════════════════════════════════════════════

document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const btnIcon = btn.querySelector('.btn-icon');

    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    btn.disabled = true;

    const facilityType = document.querySelector('input[name="facility_type"]:checked').value;

    const formData = {
        date: document.getElementById('date').value,
        occupancy: document.getElementById('occupancy').value,
        temperature: document.getElementById('temperature').value,
        is_weekend: document.getElementById('is_weekend').checked.toString(),
        is_holiday: document.getElementById('is_holiday').checked.toString(),
        exam_period: document.getElementById('exam_period').checked.toString(),
        event_flag: document.getElementById('event_flag').checked.toString(),
        prev_day_meals: document.getElementById('prev_day_meals').value,
        prev_7day_avg: document.getElementById('prev_7day_avg').value,
        weather: document.getElementById('weather').value,
        menu_type: document.getElementById('menu_type').value,
        day_of_week: document.getElementById('day_of_week').value,
        facility_type: facilityType
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            animateValue('predictionValue', 0, result.prediction, 1000);

            document.getElementById('facilityTypeDisplay').textContent =
                facilityType.charAt(0).toUpperCase() + facilityType.slice(1);

            // Pre-fill recommender with the predicted meal count
            document.getElementById('rec_predicted_meals').value = Math.round(result.prediction);

            const resultCard = document.getElementById('resultCard');
            resultCard.style.display = 'block';
            setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);

        } else {
            alert('Prediction error: ' + result.error);
        }
    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoader.style.display = 'none';
        btn.disabled = false;
    }
});

function animateValue(id, start, end, duration) {
    const el = document.getElementById(id);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        el.textContent = Math.round(current);
    }, 16);
}

// ══════════════════════════════════════════════════════════
//  TAB 2 — DISH RECOMMENDER
// ══════════════════════════════════════════════════════════

/**
 * Parse freeform inventory textarea into { ingredient: qty } object.
 * Supports:  "rice: 50"  |  "potato 20"  |  "oil=5"
 */
function parseInventoryText(text) {
    const inventory = {};
    for (const raw of text.trim().split('\n')) {
        const line = raw.trim();
        if (!line) continue;
        const parts = line.split(/[:\s=]+/);
        if (parts.length >= 2) {
            const ing = parts[0].trim().toLowerCase().replace(/\s+/g, '_');
            const qty = parseFloat(parts.slice(1).join('').replace(/[^\d.]/g, ''));
            if (ing && !isNaN(qty) && qty > 0) inventory[ing] = qty;
        }
    }
    return inventory;
}

/** Single dish result card HTML */
function renderDishCard(dish, rank) {
    const usage = Math.round(dish.inventory_usage_score * 100);
    const waste = Math.round(dish.waste_reduction_score * 100);
    const conf = Math.round(dish.confidence_score * 100);
    const menuIcon = { veg: '🥗', 'non-veg': '🍗', vegan: '🌱' }[dish.menu_type] || '🍽️';
    const medal = ['🥇', '🥈', '🥉'][rank - 1] || `<strong>#${rank}</strong>`;
    const missing = dish.missing_ingredients && dish.missing_ingredients.length
        ? `<span style="color:#fca5a5;">❌ Missing: ${dish.missing_ingredients.slice(0, 4).join(', ')}</span>`
        : `<span style="color:#6ee7b7;">✅ All ingredients available</span>`;

    return `
    <div class="dish-card" style="animation:fadeInUp .4s ease ${(rank - 1) * 0.08}s both;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem;">
            <div>
                <span style="font-size:1.35rem;">${medal}</span>
                <strong style="font-size:1.05rem;margin-left:.4rem;">${dish.dish_name}</strong>
                <span style="margin-left:.6rem;font-size:.78rem;opacity:.65;">${menuIcon} ${dish.menu_type} &nbsp;|&nbsp; ${dish.cuisine}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:.72rem;opacity:.6;">Confidence</span><br>
                <strong style="font-size:1.15rem;color:#6ee7b7;">${conf}%</strong>
            </div>
        </div>

        <div class="score-bar-wrap">
            ${bar('🟢 Inventory Usage', usage, '#6ee7b7')}
            ${bar('🟡 Waste Reduction', waste, '#f59e0b')}
        </div>

        <div class="dish-meta">
            <span>🍽️ ${dish.estimated_servings} servings</span>
            <span>⏱️ ${dish.prep_time_min} min</span>
            <span>🔥 ${dish.calories_per_serving} kcal</span>
            ${missing}
        </div>
    </div>`;
}

function bar(label, pct, color) {
    return `
    <div>
        <div class="score-bar-row">
            <span style="opacity:.72;">${label}</span>
            <span style="color:${color};font-weight:700;">${pct}%</span>
        </div>
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:${pct}%;background:${color};"></div>
        </div>
    </div>`;
}

document.getElementById('recommenderForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('recommendBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const btnIcon = btn.querySelector('.btn-icon');

    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    btn.disabled = true;

    const inventory = parseInventoryText(document.getElementById('inventoryInput').value);

    if (Object.keys(inventory).length === 0) {
        alert('Please enter at least one ingredient with a quantity.\nExample:  rice: 50');
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoader.style.display = 'none';
        btn.disabled = false;
        return;
    }

    const allergens = [];
    document.querySelectorAll('[id^="alg_"]:checked').forEach(cb => allergens.push(cb.value));

    const payload = {
        inventory,
        predicted_meals: parseInt(document.getElementById('rec_predicted_meals').value) || 200,
        menu_type: document.getElementById('rec_menu_type').value,
        top_n: parseInt(document.getElementById('rec_top_n').value) || 5,
        allergens
    };

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.success && result.recommendations.length > 0) {
            const labels = { any: 'any type', veg: 'vegetarian', 'non-veg': 'non-vegetarian', vegan: 'vegan' };
            document.getElementById('recResultSubtitle').textContent =
                `Top ${result.recommendations.length} ${labels[payload.menu_type] || payload.menu_type} dishes · ${payload.predicted_meals} expected meals`;

            document.getElementById('recResultList').innerHTML =
                result.recommendations.map((d, i) => renderDishCard(d, i + 1)).join('');

            const card = document.getElementById('recResultCard');
            card.style.display = 'block';
            setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);

        } else if (result.success) {
            alert('No matching dishes found.\nTry changing the menu type or reducing the minimum match threshold.');
        } else {
            alert('Recommendation error: ' + result.error);
        }

    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoader.style.display = 'none';
        btn.disabled = false;
    }
});
