const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const traceSidebar = document.getElementById('trace-sidebar');
const traceContent = document.getElementById('trace-content');
const traceToggleBtn = document.querySelector('.toggle-btn');
const traceHeader = document.getElementById('trace-toggle');

// Toggle Sidebar
function toggleSidebar() {
    traceSidebar.classList.toggle('open');
    const icon = traceToggleBtn.querySelector('i');
    // Simple rotation logic via CSS class or just icon swap
    // The CSS handles rotation for .toggle-btn inside .open
}

traceHeader.addEventListener('click', toggleSidebar);

// Send Message
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add User Message
    addMessage(text, 'user');
    userInput.value = '';
    
    // Hide welcome message if it's the first message
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.style.display = 'none';

    // 2. Show Loading
    const loadingId = addLoadingMessage();

    try {
        // 3. Call API
        // Assuming backend is running on port 8000
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        });

        if (!response.ok) throw new Error('Network response was not ok');
        
        const data = await response.json();
        
        // 4. Remove Loading
        removeMessage(loadingId);

        // 5. Add Agent Message
        addMessage(data.answer, 'agent', data.trace);

        // 6. Update Trace (but don't auto-open, let user click)
        renderTrace(data.trace);

    } catch (error) {
        removeMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'agent');
    }
}

function addMessage(text, sender, traceData = null) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'content';
    
    // Render Markdown for agent
    if (sender === 'agent') {
        content.innerHTML = marked.parse(text);
        
        if (traceData && traceData.length > 0) {
            const traceBtn = document.createElement('button');
            traceBtn.className = 'view-trace-btn';
            traceBtn.innerHTML = '<i class="fa-solid fa-code-branch"></i> View Thought Process';
            traceBtn.onclick = () => {
                renderTrace(traceData);
                if (!traceSidebar.classList.contains('open')) {
                    toggleSidebar();
                }
            };
            // Inline styles for the button
            traceBtn.style.marginTop = '15px';
            traceBtn.style.background = 'rgba(255,255,255,0.05)';
            traceBtn.style.border = '1px solid rgba(255,255,255,0.1)';
            traceBtn.style.padding = '8px 12px';
            traceBtn.style.color = '#a8a8a8';
            traceBtn.style.borderRadius = '6px';
            traceBtn.style.cursor = 'pointer';
            traceBtn.style.fontSize = '0.85rem';
            traceBtn.style.display = 'flex';
            traceBtn.style.alignItems = 'center';
            traceBtn.style.gap = '8px';
            traceBtn.style.transition = 'background 0.2s';
            
            traceBtn.onmouseover = () => traceBtn.style.background = 'rgba(255,255,255,0.1)';
            traceBtn.onmouseout = () => traceBtn.style.background = 'rgba(255,255,255,0.05)';
            
            content.appendChild(traceBtn);
        }
    } else {
        content.textContent = text;
    }
    
    div.appendChild(avatar);
    div.appendChild(content);
    
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message agent';
    div.id = id;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'content';
    content.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Thinking...';
    
    div.appendChild(avatar);
    div.appendChild(content);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function renderTrace(trace) {
    traceContent.innerHTML = '';
    if (!trace || trace.length === 0) {
        traceContent.innerHTML = '<div class="empty-trace">No trace available.</div>';
        return;
    }

    trace.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = `trace-item ${item.type}`;
        
        const header = document.createElement('div');
        header.className = 'trace-item-header';
        
        // Icon selection
        let icon = '<i class="fa-solid fa-circle-info"></i>';
        let title = item.type;
        
        if (item.type === 'model') {
            icon = '<i class="fa-solid fa-brain"></i>';
            title = 'Thinking';
        } else if (item.type === 'tool') {
            icon = '<i class="fa-solid fa-screwdriver-wrench"></i>';
            title = `Tool: ${item.action || 'Unknown'}`;
        } else if (item.type === 'consistency_vote') {
            icon = '<i class="fa-solid fa-check-double"></i>';
            title = 'Consistency Check';
        }

        header.innerHTML = `${icon} <span>${title}</span> <span style="opacity:0.5; font-size:0.8em">#${index + 1}</span>`;
        
        const body = document.createElement('div');
        body.className = 'trace-item-body';
        
        // --- RENDER LOGIC ---
        
        // 1. Model Thought
        if (item.type === 'model') {
            body.innerHTML = `<div class="thought-text">${item.thought || item.content || ''}</div>`;
        } 
        
        // 2. Tool Execution
        else if (item.type === 'tool') {
            // Input Args
            let argsDisplay = item.args;
            try {
                if (typeof item.args === 'string') argsDisplay = JSON.parse(item.args);
            } catch(e) {}
            
            const inputHtml = `<div class="tool-input">
                <span class="label">Input:</span> <code>${JSON.stringify(argsDisplay)}</code>
            </div>`;

            // Output / Observation
            let outputHtml = '';
            let obs = item.observation;
            
            // Try to parse observation as JSON
            let obsObj = null;
            try {
                obsObj = JSON.parse(obs);
            } catch(e) {}

            if (obsObj && obsObj.results && Array.isArray(obsObj.results)) {
                // --- Special Rendering for Search Results ---
                outputHtml += `<div class="tool-output search-results">
                    <span class="label">Search Results:</span>
                    <div class="search-list">`;
                
                obsObj.results.forEach(res => {
                    outputHtml += `
                        <div class="search-card">
                            <div class="search-title">
                                <a href="${res.url}" target="_blank">${res.title || 'No Title'}</a>
                                <span class="search-source">[${res.doc_id || res.source}]</span>
                            </div>
                            <div class="search-snippet">${res.snippet || ''}</div>
                            ${res.score ? `<div class="search-meta">Score: ${res.score}</div>` : ''}
                        </div>`;
                });
                outputHtml += `</div></div>`;
            } 
            else if (obsObj && obsObj.plan && Array.isArray(obsObj.plan)) {
                // --- Special Rendering for Planner ---
                outputHtml += `<div class="tool-output plan-results">
                    <span class="label">Plan Generated:</span>
                    <ul class="plan-list">`;
                obsObj.plan.forEach(step => {
                    outputHtml += `<li><strong>Step ${step.step}:</strong> ${step.action}</li>`;
                });
                outputHtml += `</ul></div>`;
            }
            else {
                // --- Generic Output ---
                // If it's a long string, maybe truncate or scroll
                outputHtml = `<div class="tool-output">
                    <span class="label">Output:</span>
                    <pre>${obs}</pre>
                </div>`;
            }

            body.innerHTML = inputHtml + outputHtml;
        } 
        
        // 3. Consistency Vote
        else if (item.type === 'consistency_vote') {
             body.innerHTML = `
                <div><strong>Method:</strong> ${item.method}</div>
                <div><strong>Winner Count:</strong> ${item.winner_count || 'N/A'}</div>
             `;
        } 
        else {
            // Fallback
            body.textContent = JSON.stringify(item, null, 2);
        }
        
        div.appendChild(header);
        div.appendChild(body);
        traceContent.appendChild(div);
    });
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
