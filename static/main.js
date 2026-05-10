document.addEventListener('DOMContentLoaded', () => {
    const qaForm = document.getElementById('qa-form');
    const modelTypeSelect = document.getElementById('model-type');
    const extSettings = document.getElementById('extractive-settings');
    const genSettings = document.getElementById('generative-settings');
    const accordionHeader = document.getElementById('advanced-settings-toggle');
    const resultContainer = document.getElementById('result-container');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const answerText = document.getElementById('answer-text');
    const statsContainer = document.getElementById('stats-container');
    const submitBtn = document.getElementById('submit-btn');

    let abortController = null;

    // Toggle Advanced Settings Accordion
    accordionHeader.addEventListener('click', () => {
        accordionHeader.classList.toggle('active');
    });

    // Update range input displays
    const ranges = document.querySelectorAll('input[type="range"]');
    ranges.forEach(range => {
        const display = document.getElementById(`${range.id}-val`);
        range.addEventListener('input', () => {
            display.textContent = range.value;
        });
    });

    // Toggle settings based on model type
    modelTypeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'extractive') {
            extSettings.classList.remove('hidden');
            genSettings.classList.add('hidden');


#     const genSettings = document.getElementById('generative-settings');
#     const accordionHeader = document.getElementById('advanced-settings-toggle');
#     const resultContainer = document.getElementById('result-container');
#     const loader = document.getElementById('loader');
#     const resultContent = document.getElementById('result-content');
#     const answerText = document.getElementById('answer-text');
#     const statsContainer = document.getElementById('stats-container');
#     const submitBtn = document.getElementById('submit-btn');
# 
#     let abortController = null;
# 
#     // Toggle Advanced Settings Accordion
#     accordionHeader.addEventListener('click', () => {
#         accordionHeader.classList.toggle('active');
#     });
# 
#     // Update range input displays
#     const ranges = document.querySelectorAll('input[type="range"]');
#     ranges.forEach(range => {
#         const display = document.getElementById(`${range.id}-val`);
#         range.addEventListener('input', () => {
#             display.textContent = range.value;
#         });
#     });
# 
#     // Toggle settings based on model type
#     modelTypeSelect.addEventListener('change', (e) => {
#         if (e.target.value === 'extractive') {
#             extSettings.classList.remove('hidden');
#             genSettings.classList.add('hidden');
