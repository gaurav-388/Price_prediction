let priceChart = null;

document.getElementById('btn').addEventListener('click', async function () {
  const district = document.getElementById('district').value;
  const date = document.getElementById('date').value;
  const result = document.getElementById('result');
  result.innerHTML = 'Loading...';

  if (!date) {
    result.innerHTML = `<div class="alert alert-danger">Please select a date.</div>`;
    return;
  }

  const resp = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ district, date, horizon: 7 })
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: 'Unknown error' }));
    result.innerHTML = `<div class="alert alert-danger">${err.error || 'Request failed'}</div>`;
    return;
  }

  const js = await resp.json();
  let html = `<h4>Forecast for ${js.district} — next 7 days</h4>`;
  html += '<table class="table table-sm"><thead><tr><th>Date</th><th>Predicted Price</th></tr></thead><tbody>';
  js.forecast.forEach(r => {
    html += `<tr><td>${r.date}</td><td>${r.predicted_price.toFixed(2)}</td></tr>`;
  });
  html += '</tbody></table>';
  result.innerHTML = html;

  // Chart
  const ctx = document.getElementById('priceChart').getContext('2d');
  if (priceChart) {
    priceChart.destroy();
  }
  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: js.forecast.map(r => r.date),
      datasets: [{
        label: 'Predicted Price',
        data: js.forecast.map(r => r.predicted_price),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
  });
});
