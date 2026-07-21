// CHANGE THIS to your backend URL
const BACKEND_URL = "http://127.0.0.1:8000"; // e.g. Streamlit or FastAPI

document.getElementById("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const input = document.getElementById("question");
  const question = input.value.trim();
  if (!question) return;

  addMessage("user", question);
  input.value = "";
  setInputEnabled(false);

  try {
    // Get current tab URL
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = new URL(tab.url);

    if (!url.hostname.includes("youtube.com")) {
      addMessage("bot", "Please open a YouTube video first.");
      setInputEnabled(true);
      return;
    }

    const videoId = url.searchParams.get("v");
    if (!videoId) {
      addMessage("bot", "Could not detect video ID from this page.");
      setInputEnabled(true);
      return;
    }

    const res = await fetch(`${BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, question }),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    const answer = data.answer || "No answer returned from backend.";
    addMessage("bot", answer);
  } catch (err) {
    console.error(err);
    addMessage("bot", "Error contacting chatbot backend. Is it running?");
  } finally {
    setInputEnabled(true);
  }
});

function addMessage(role, text) {
  const container = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function setInputEnabled(enabled) {
  const input = document.getElementById("question");
  const button = document.querySelector("#chat-form button");
  input.disabled = !enabled;
  button.disabled = !enabled;
  if (enabled) input.focus();
}