/* MAI-DxO Simple Interface - Debug Version */

console.log('app_simple.js loading...');

// Global variables (not in IIFE for easier debugging)
let currentSocket = null;
let selectedModel = 'gpt-4o';

// Demo case data
const DEMO_CASE = {
  initial: "A 29-year-old woman was admitted to the hospital because of sore throat and peritonsillar swelling and bleeding.",
  fullCase: "29-year-old female with peritonsillar mass",
  groundTruth: "Rhabdomyosarcoma"
};

// Simple message function
function addMessage(sender, content) {
  console.log(`Message from ${sender}: ${content}`);
  const messagesArea = document.getElementById('messages');
  if (!messagesArea) {
    console.error('Messages area not found!');
    return;
  }
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `message-wrapper ${sender}`;
  messageDiv.innerHTML = `
    <div class="speaker-label">${sender}</div>
    <div class="message-bubble">${content}</div>
  `;
  messagesArea.appendChild(messageDiv);
}

// API Key Modal Functions
function openApiKeyModal() {
  console.log('Opening API key modal...');
  const modal = document.getElementById('api-key-modal');
  if (modal) {
    modal.classList.remove('hidden');
    console.log('Modal opened');
  } else {
    console.error('API key modal not found!');
    alert('API key modal not found in HTML!');
  }
}

function closeApiKeyModal() {
  console.log('Closing API key modal...');
  const modal = document.getElementById('api-key-modal');
  if (modal) {
    modal.classList.add('hidden');
  }
}

async function saveApiKeys() {
  console.log('Saving API keys...');
  const openaiKey = document.getElementById('openai-key-input')?.value || '';
  const anthropicKey = document.getElementById('anthropic-key-input')?.value || '';
  const geminiKey = document.getElementById('gemini-key-input')?.value || '';

  const payload = {
    "OPENAI_API_KEY": openaiKey,
    "ANTHROPIC_API_KEY": anthropicKey,
    "GEMINI_API_KEY": geminiKey
  };

  try {
    const response = await fetch('/api/save_keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (response.ok) {
      alert('API keys saved successfully!');
      closeApiKeyModal();
    } else {
      alert('Failed to save API keys');
    }
  } catch (error) {
    console.error('Error saving keys:', error);
    alert('Error saving API keys: ' + error.message);
  }
}

// Start diagnostic session
async function startDiagnosticSession() {
  console.log('Starting diagnostic session...');
  
  const caseInput = document.getElementById('case-input');
  const modeRadio = document.querySelector('input[name="diagnostic-mode"]:checked');
  const isAutonomous = modeRadio?.value === 'autonomous';
  
  let text = caseInput?.value?.trim() || '';
  
  if (!text && !isAutonomous) {
    alert('Please enter a patient case or select Autonomous Demo');
    return;
  }
  
  if (isAutonomous) {
    text = DEMO_CASE.initial;
  }
  
  const payload = {
    initial_presentation: text,
    full_case_details: isAutonomous ? DEMO_CASE.fullCase : '',
    ground_truth: isAutonomous ? DEMO_CASE.groundTruth : '',
    model_name: selectedModel,
    max_iterations: 5,
    mode: 'no_budget',
    interactive: !isAutonomous
  };
  
  console.log('Sending request:', payload);
  
  try {
    const response = await fetch('/api/start_case', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    console.log('Case started:', result);
    
    // Connect WebSocket
    connectWebSocket(result.case_id);
    
    // Show message
    addMessage('system', '🚀 Diagnostic session started');
    addMessage('clinician', text);
    
  } catch (error) {
    console.error('Failed to start case:', error);
    alert('Failed to start diagnostic session: ' + error.message);
  }
}

// WebSocket connection
function connectWebSocket(caseId) {
  console.log('Connecting WebSocket for case:', caseId);
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${protocol}://${location.host}/ws/${caseId}`);
  
  currentSocket = socket;
  
  socket.onopen = () => {
    console.log('WebSocket connected');
    addMessage('system', '✅ Connected to diagnostic engine');
  };
  
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log('WebSocket message:', data);
      handleWebSocketMessage(data);
    } catch (error) {
      console.error('WebSocket message error:', error);
    }
  };
  
  socket.onerror = (error) => {
    console.error('WebSocket error:', error);
    addMessage('system', '❌ Connection error');
  };
  
  socket.onclose = () => {
    console.log('WebSocket closed');
    addMessage('system', 'Session ended');
    currentSocket = null;
  };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
  switch (data.type) {
    case 'log':
      console.log('Log:', data.message);
      break;
    case 'pause':
      console.log('Pause for input:', data.action);
      const action = data.action;
      if (action?.action_type === 'ask') {
        addMessage('mai-dxo', `Question: ${action.content}`);
      } else if (action?.action_type === 'test') {
        addMessage('mai-dxo', `Tests ordered: ${action.content}`);
      }
      // Show input field
      const inputContainer = document.getElementById('input-container');
      if (inputContainer) {
        inputContainer.style.display = 'flex';
      }
      break;
    case 'result':
      addMessage('mai-dxo', `Final Diagnosis: ${data.final_diagnosis}`);
      if (data.accuracy_score) {
        addMessage('system', `Accuracy: ${data.accuracy_score}/5.0`);
      }
      break;
    case 'error':
      addMessage('system', `❌ Error: ${data.message}`);
      break;
    case 'done':
      console.log('Session complete');
      break;
    default:
      console.log('Unknown message type:', data.type);
  }
}

// Load demo case
function loadDemoCase() {
  console.log('Loading demo case...');
  const caseInput = document.getElementById('case-input');
  if (caseInput) {
    caseInput.value = DEMO_CASE.initial;
  }
  
  // Set to autonomous mode
  const autonomousRadio = document.querySelector('input[value="autonomous"]');
  if (autonomousRadio) {
    autonomousRadio.checked = true;
  }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, initializing...');
  
  // API Key button
  const apiKeyBtn = document.getElementById('api-key-btn');
  if (apiKeyBtn) {
    console.log('Found API key button, attaching listener');
    apiKeyBtn.onclick = openApiKeyModal;
  } else {
    console.error('API key button not found!');
  }
  
  // API Modal buttons
  const apiSaveBtn = document.getElementById('api-key-save');
  if (apiSaveBtn) {
    apiSaveBtn.onclick = saveApiKeys;
  }
  
  const apiCancelBtn = document.getElementById('api-key-cancel');
  if (apiCancelBtn) {
    apiCancelBtn.onclick = closeApiKeyModal;
  }
  
  const modalCloseBtn = document.querySelector('#api-key-modal .modal-close');
  if (modalCloseBtn) {
    modalCloseBtn.onclick = closeApiKeyModal;
  }
  
  // Start button
  const startBtn = document.getElementById('start-case-btn');
  if (startBtn) {
    console.log('Found start button, attaching listener');
    startBtn.onclick = startDiagnosticSession;
  } else {
    console.error('Start button not found!');
  }
  
  // Load demo button
  const loadDemoBtn = document.getElementById('load-demo-btn');
  if (loadDemoBtn) {
    loadDemoBtn.onclick = loadDemoCase;
  }
  
  // Send button for interactive mode
  const sendBtn = document.getElementById('send-button');
  if (sendBtn) {
    sendBtn.onclick = function() {
      const userInput = document.getElementById('user-input');
      if (userInput && currentSocket) {
        const text = userInput.value.trim();
        if (text) {
          currentSocket.send(JSON.stringify({ type: 'input', data: text }));
          addMessage('clinician', text);
          userInput.value = '';
          document.getElementById('input-container').style.display = 'none';
        }
      }
    };
  }
  
  console.log('✅ MAI-DxO Simple Interface initialized');
});

console.log('app_simple.js loaded successfully');
