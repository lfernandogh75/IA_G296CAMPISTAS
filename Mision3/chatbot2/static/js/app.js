const chat = document.getElementById('chat');
const input = document.getElementById('input');
const btn = document.getElementById('send');

function addMsg(text, who = 'bot') {
  const div = document.createElement('div');
  div.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  input.value = '';
  btn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    addMsg(data.reply || "(sin respuesta)");
  } catch (e) {
    addMsg("Error de red: " + e.toString());
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

btn.addEventListener('click', send);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });

addMsg("¡Hola! Soy tu asistente. ¿En qué puedo ayudarte hoy?");
