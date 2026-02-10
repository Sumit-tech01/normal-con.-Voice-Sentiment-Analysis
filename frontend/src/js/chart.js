/**
 * Chart Manager - Chart.js visualizations for emotion analysis
 */

// Chart instance
let chart = null;

// Chart colors
const COLORS = {
  joy: '#22c55e',
  sadness: '#3b82f6',
  anger: '#ef4444',
  fear: '#a855f7',
  love: '#ec4899',
  surprise: '#f59e0b',
  neutral: '#6b7280',
};

/**
 * Initialize the emotions chart
 */
export function initChart() {
  const canvas = document.getElementById('emotions-chart');
  if (!canvas) {
    console.warn('Chart canvas not found');
    return;
  }

  const ctx = canvas.getContext('2d');

  // Initial data
  const initialData = {
    labels: ['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise'],
    datasets: [{
      data: [0, 0, 0, 0, 0, 0],
      backgroundColor: [
        COLORS.joy + '80',
        COLORS.sadness + '80',
        COLORS.anger + '80',
        COLORS.fear + '80',
        COLORS.love + '80',
        COLORS.surprise + '80',
      ],
      borderColor: [
        COLORS.joy,
        COLORS.sadness,
        COLORS.anger,
        COLORS.fear,
        COLORS.love,
        COLORS.surprise,
      ],
      borderWidth: 2,
      hoverBackgroundColor: Object.values(COLORS),
    }]
  };

  chart = new Chart(ctx, {
    type: 'doughnut',
    data: initialData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleColor: '#fff',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(168, 85, 247, 0.3)',
          borderWidth: 1,
          padding: 12,
          displayColors: true,
          callbacks: {
            label: (context) => {
              return `${context.label}: ${(context.raw * 100).toFixed(1)}%`;
            }
          }
        }
      },
      animation: {
        animateRotate: true,
        animateScale: true,
        duration: 500,
        easing: 'easeOutQuart'
      }
    }
  });

  console.log('Chart initialized');
}

/**
 * Update chart with new emotion data
 * @param {Object} emotions - Emotion scores object
 */
export function updateChart(emotions) {
  if (!chart) {
    console.warn('Chart not initialized');
    return;
  }

  const orderedEmotions = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise'];
  const data = orderedEmotions.map(e => emotions[e] || 0);

  chart.data.datasets[0].data = data;
  chart.update('active');
}

/**
 * Reset chart to empty state
 */
export function resetChart() {
  if (!chart) return;

  chart.data.datasets[0].data = [0, 0, 0, 0, 0, 0];
  chart.update();
}

/**
 * Get chart instance
 */
export function getChart() {
  return chart;
}

/**
 * Destroy chart instance
 */
export function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
}
