// Food Waste Prediction - Enhanced Multi-Facility UI

// Set default date to today
document.getElementById('date').valueAsDate = new Date();

// Update occupancy display
document.getElementById('occupancy').addEventListener('input', function () {
    document.getElementById('occupancy-value').textContent = this.value;
});

// Facility type selection handler
const facilityRadios = document.querySelectorAll('input[name="facility_type"]');
const facilityTypeLabel = document.getElementById('facilityTypeLabel');

facilityRadios.forEach(radio => {
    radio.addEventListener('change', function () {
        updateFacilityLabels(this.value);
    });
});

function updateFacilityLabels(facilityType) {
    // Update the section label dynamically
    if (facilityType === 'hostel') {
        facilityTypeLabel.textContent = 'Hostel';
    } else if (facilityType === 'restaurant') {
        facilityTypeLabel.textContent = 'Restaurant';
    }
}

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    const btnIcon = btn.querySelector('.btn-icon');

    // Show loading
    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    btn.disabled = true;
    btn.classList.add('loading');

    // Get selected facility type
    const facilityType = document.querySelector('input[name="facility_type"]:checked').value;

    // Collect form data
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
        meals_prepared: document.getElementById('meals_prepared').value,
        weather: document.getElementById('weather').value,
        menu_type: document.getElementById('menu_type').value,
        day_of_week: document.getElementById('day_of_week').value,
        facility_type: facilityType
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (result.success) {
            // Animate number counting
            animateValue('predictionValue', 0, result.prediction, 1000);

            // Update facility type display
            const facilityDisplay = document.getElementById('facilityTypeDisplay');
            facilityDisplay.textContent = facilityType.charAt(0).toUpperCase() + facilityType.slice(1);

            // Show result card
            const resultCard = document.getElementById('resultCard');
            resultCard.style.display = 'block';

            // Scroll to result
            setTimeout(() => {
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    } finally {
        // Hide loading
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoader.style.display = 'none';
        btn.disabled = false;
        btn.classList.remove('loading');
    }
});

// Animate number counting
function animateValue(id, start, end, duration) {
    const element = document.getElementById(id);
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}
