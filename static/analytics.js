function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartAxisOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false }, ticks: { color: cssVar("--chart-tick"), font: { family: "Inter", size: 11 } } },
      y: { beginAtZero: true, ticks: { color: cssVar("--chart-tick"), stepSize: 1, font: { family: "Inter", size: 11 } }, grid: { color: cssVar("--chart-grid") } }
    }
  };
}

let trendChart = null;
let personChart = null;
let weekdayChart = null;

async function refreshTrendChart() {
  try {
    const res = await fetch("/api/attendance/trends");
    const data = await res.json();
    const labels = data.map(d => d.date.slice(5));
    const values = data.map(d => d.present);

    const ctx = document.getElementById("trendChart").getContext("2d");
    if (trendChart) {
      trendChart.data.labels = labels;
      trendChart.data.datasets[0].data = values;
      trendChart.update();
      return;
    }

    trendChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          data: values,
          borderColor: cssVar("--blue"),
          backgroundColor: cssVar("--blue") + "14",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          pointBackgroundColor: cssVar("--blue"),
        }]
      },
      options: chartAxisOptions()
    });
  } catch (err) {
    console.error("Failed to refresh trend chart:", err);
  }
}

async function refreshPersonChart() {
  try {
    const res = await fetch("/api/attendance/summary");
    const data = await res.json();
    const labels = data.map(d => d.name);
    const values = data.map(d => d.days);

    const ctx = document.getElementById("personChart").getContext("2d");
    if (personChart) {
      personChart.data.labels = labels;
      personChart.data.datasets[0].data = values;
      personChart.update();
      return;
    }

    personChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ data: values, backgroundColor: cssVar("--green"), borderRadius: 4, maxBarThickness: 26 }]
      },
      options: chartAxisOptions()
    });
  } catch (err) {
    console.error("Failed to refresh person chart:", err);
  }
}

async function refreshWeekdayChart() {
  try {
    const res = await fetch("/api/attendance/weekday-pattern");
    const data = await res.json();
    const labels = data.map(d => d.day);
    const values = data.map(d => d.average_present);

    const ctx = document.getElementById("weekdayChart").getContext("2d");
    if (weekdayChart) {
      weekdayChart.data.datasets[0].data = values;
      weekdayChart.update();
      return;
    }

    weekdayChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ data: values, backgroundColor: cssVar("--navy"), borderRadius: 4, maxBarThickness: 34 }]
      },
      options: chartAxisOptions()
    });
  } catch (err) {
    console.error("Failed to refresh weekday chart:", err);
  }
}

window.refreshChartTheme = function () {
  if (trendChart) {
    trendChart.data.datasets[0].borderColor = cssVar("--blue");
    trendChart.data.datasets[0].backgroundColor = cssVar("--blue") + "14";
    trendChart.data.datasets[0].pointBackgroundColor = cssVar("--blue");
    trendChart.options = chartAxisOptions();
    trendChart.update();
  }
  if (personChart) {
    personChart.data.datasets[0].backgroundColor = cssVar("--green");
    personChart.options = chartAxisOptions();
    personChart.update();
  }
  if (weekdayChart) {
    weekdayChart.data.datasets[0].backgroundColor = cssVar("--navy");
    weekdayChart.options = chartAxisOptions();
    weekdayChart.update();
  }
};

refreshTrendChart();
refreshPersonChart();
refreshWeekdayChart();
setInterval(refreshTrendChart, 20000);
setInterval(refreshPersonChart, 20000);
setInterval(refreshWeekdayChart, 30000);
