/**
 * Handles client‑side interaction for the Dagen reels generator.
 *
 * This script reads the article URL from the input field, displays a
 * status message, and calls a placeholder backend endpoint to
 * generate the video.  Replace the fetch call with your actual API
 * endpoint that triggers the Python pipeline on the server.
 */

document.addEventListener('DOMContentLoaded', () => {
  const generateBtn = document.getElementById('generateBtn');
  const articleUrl = document.getElementById('articleUrl');
  const statusDiv = document.getElementById('status');
  const resultDiv = document.getElementById('result');

  generateBtn.addEventListener('click', async () => {
    const url = articleUrl.value.trim();
    if (!url) {
      statusDiv.textContent = 'Vennligst lim inn en gyldig artikkel‑URL.';
      return;
    }
    // Disable UI while processing
    generateBtn.disabled = true;
    statusDiv.textContent = 'Genererer video … dette kan ta noen minutter.';
    resultDiv.textContent = '';

    try {
      // Placeholder: call backend endpoint
      // Replace '/api/convert' with the actual endpoint on your server
      const response = await fetch('/api/convert?url=' + encodeURIComponent(url));
      if (!response.ok) {
        throw new Error('Serverfeil: ' + response.status);
      }
      const data = await response.json();
      // Expecting data.videoUrl and optionally data.subtitleUrl
      statusDiv.textContent = 'Video generert!';
      const videoLink = document.createElement('a');
      videoLink.href = data.videoUrl;
      videoLink.textContent = 'Last ned video';
      videoLink.target = '_blank';
      resultDiv.appendChild(videoLink);
      if (data.subtitleUrl) {
        const subtitleLink = document.createElement('a');
        subtitleLink.href = data.subtitleUrl;
        subtitleLink.textContent = 'Last ned undertekster';
        subtitleLink.target = '_blank';
        subtitleLink.style.display = 'block';
        resultDiv.appendChild(subtitleLink);
      }
    } catch (err) {
      console.error(err);
      statusDiv.textContent = 'Det oppstod en feil ved generering av video.';
    } finally {
      generateBtn.disabled = false;
    }
  });
});