/**
 * WebScout API Documentation - API Interaction Functions
 * Handles all API requests and responses
 */

// API interaction utilities
window.WebScoutAPI = {
    baseUrl: '',
    
    init(baseUrl) {
        this.baseUrl = baseUrl || window.WEBSCOUT_CONFIG?.baseUrl || '';
    },
    
    // Generic API request function
    async makeRequest(endpoint, method = 'GET', body = null, requireAuth = true) {
        const url = `${this.baseUrl}${endpoint}`;
        const startTime = Date.now();
        
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (requireAuth && WebScoutApp.currentApiKey) {
            headers['Authorization'] = `Bearer ${WebScoutApp.currentApiKey}`;
        }
        
        const options = {
            method,
            headers
        };
        
        if (body && method !== 'GET') {
            options.body = JSON.stringify(body);
        }
        
        try {
            const response = await fetch(url, options);
            const endTime = Date.now();
            const responseTime = endTime - startTime;
            
            let data;
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }
            
            return {
                status: response.status,
                statusText: response.statusText,
                data,
                responseTime,
                headers: Object.fromEntries(response.headers.entries())
            };
        } catch (error) {
            return {
                status: 0,
                statusText: 'Network Error',
                data: { error: error.message },
                responseTime: Date.now() - startTime,
                headers: {}
            };
        }
    },
    
    // Streaming request handler
    async makeStreamingRequest(endpoint, body, onChunk, onComplete, onError) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (WebScoutApp.currentApiKey) {
            headers['Authorization'] = `Bearer ${WebScoutApp.currentApiKey}`;
        }
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers,
                body: JSON.stringify(body)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                    onComplete?.();
                    break;
                }
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            onComplete?.();
                            return;
                        }
                        
                        try {
                            const parsed = JSON.parse(data);
                            onChunk?.(parsed);
                        } catch (e) {
                            // Skip invalid JSON
                        }
                    }
                }
            }
        } catch (error) {
            onError?.(error);
        }
    }
};

// API test functions
async function testChatCompletion(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);
    
    const model = document.getElementById('chat-model').value;
    const messagesText = document.getElementById('chat-messages').value;
    const temperature = parseFloat(document.getElementById('chat-temperature').value);
    const maxTokens = parseInt(document.getElementById('chat-max-tokens').value);
    const stream = document.getElementById('chat-stream').checked;
    
    let messages;
    try {
        messages = JSON.parse(messagesText || '[{"role": "user", "content": "Hello!"}]');
    } catch (e) {
        WebScoutApp.showToast('Invalid JSON in messages field', 'error');
        hideLoading(button, originalText);
        return;
    }
    
    const body = {
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream
    };
    
    if (stream) {
        await handleStreamingChatCompletion(body, button, originalText);
    } else {
        const response = await WebScoutAPI.makeRequest('/v1/chat/completions', 'POST', body);
        displayResponse('chat-response', response);
        hideLoading(button, originalText);
    }
}

async function handleStreamingChatCompletion(body, button, originalText) {
    const responseDiv = document.getElementById('chat-response');
    responseDiv.innerHTML = '';
    responseDiv.classList.add('show');
    
    const statusDiv = document.createElement('div');
    statusDiv.className = 'response-header';
    statusDiv.innerHTML = `
        <div class="response-status">
            <span class="status-code status-200">STREAMING</span>
            <span class="response-time">Connecting...</span>
        </div>
    `;
    responseDiv.appendChild(statusDiv);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'response-body';
    const preElement = document.createElement('pre');
    contentDiv.appendChild(preElement);
    responseDiv.appendChild(contentDiv);
    
    let fullContent = '';
    const startTime = Date.now();
    
    await WebScoutAPI.makeStreamingRequest(
        '/v1/chat/completions',
        body,
        (chunk) => {
            if (chunk.choices && chunk.choices[0]?.delta?.content) {
                const content = chunk.choices[0].delta.content;
                fullContent += content;
                preElement.textContent = fullContent;
            }
        },
        () => {
            const endTime = Date.now();
            statusDiv.querySelector('.response-time').textContent = `${endTime - startTime}ms`;
            hideLoading(button, originalText);
        },
        (error) => {
            preElement.textContent = `Error: ${error.message}`;
            statusDiv.querySelector('.status-code').className = 'status-code status-500';
            statusDiv.querySelector('.status-code').textContent = 'ERROR';
            hideLoading(button, originalText);
        }
    );
}

async function testImageGeneration(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);
    
    const prompt = document.getElementById('image-prompt').value;
    const model = document.getElementById('image-model').value;
    const size = document.getElementById('image-size').value;
    const n = parseInt(document.getElementById('image-count').value);
    
    if (!prompt.trim()) {
        WebScoutApp.showToast('Please enter a prompt', 'error');
        hideLoading(button, originalText);
        return;
    }
    
    const body = { prompt, model, size, n };
    const response = await WebScoutAPI.makeRequest('/v1/images/generations', 'POST', body);
    
    displayImageResponse('image-response', response);
    hideLoading(button, originalText);
}

async function testListModels(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const response = await WebScoutAPI.makeRequest('/v1/models', 'GET');
    displayResponse('models-response', response);
    hideLoading(button, originalText);
}

async function testListTTIModels(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const response = await WebScoutAPI.makeRequest('/v1/TTI/models', 'GET');
    displayResponse('tti-models-response', response);
    hideLoading(button, originalText);
}

async function testGetModel(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);
    
    const modelId = document.getElementById('model-id-input').value;
    
    if (!modelId.trim()) {
        WebScoutApp.showToast('Please enter a model ID', 'error');
        hideLoading(button, originalText);
        return;
    }
    
    const response = await WebScoutAPI.makeRequest(`/v1/models/${modelId}`, 'GET');
    displayResponse('model-info-response', response);
    hideLoading(button, originalText);
}

async function testGenerateApiKey(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const username = document.getElementById('auth-username').value;
    const name = document.getElementById('auth-name').value;
    const rateLimit = parseInt(document.getElementById('auth-rate-limit').value);
    const expiresInDays = parseInt(document.getElementById('auth-expires').value) || null;

    if (!username.trim()) {
        WebScoutApp.showToast('Please enter a username', 'error');
        hideLoading(button, originalText);
        return;
    }

    const body = {
        username,
        name: name || null,
        rate_limit: rateLimit,
        expires_in_days: expiresInDays
    };
    const response = await WebScoutAPI.makeRequest('/v1/auth/generate-key', 'POST', body, false);

    if (response.status === 200 && response.data.api_key) {
        WebScoutApp.currentApiKey = response.data.api_key;
        localStorage.setItem('webscout_api_key', WebScoutApp.currentApiKey);
        WebScoutApp.showToast('API key generated and saved!', 'success');
    }

    displayResponse('auth-response', response);
    hideLoading(button, originalText);
}

async function testValidateApiKey(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const apiKey = document.getElementById('validate-api-key').value;

    if (!apiKey.trim()) {
        WebScoutApp.showToast('Please enter an API key', 'error');
        hideLoading(button, originalText);
        return;
    }

    // Temporarily use the provided key for validation
    const originalKey = WebScoutApp.currentApiKey;
    WebScoutApp.currentApiKey = apiKey;

    const response = await WebScoutAPI.makeRequest('/v1/auth/validate', 'GET');

    // Restore original key
    WebScoutApp.currentApiKey = originalKey;

    displayResponse('validate-response', response);
    hideLoading(button, originalText);
}

async function testWebSearch(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const query = document.getElementById('search-query').value;
    const engine = document.getElementById('search-engine').value;
    const type = document.getElementById('search-type').value;
    const maxResults = parseInt(document.getElementById('search-max-results').value);
    const region = document.getElementById('search-region').value;
    const safesearch = document.getElementById('search-safesearch').value;

    if (!query.trim()) {
        WebScoutApp.showToast('Please enter a search query', 'error');
        hideLoading(button, originalText);
        return;
    }

    const params = new URLSearchParams({
        q: query,
        engine,
        type,
        max_results: maxResults,
        region,
        safesearch
    });

    const response = await WebScoutAPI.makeRequest(`/search?${params}`, 'GET', null, false);
    displayResponse('search-response', response);
    hideLoading(button, originalText);
}

async function testHealthCheck(event) {
    const button = event?.target || event;
    const originalText = showLoading(button);

    const response = await WebScoutAPI.makeRequest('/health', 'GET', null, false);
    displayResponse('health-response', response);
    hideLoading(button, originalText);
}

// Load models for the models tab
async function loadModels() {
    const container = document.getElementById('models-container');
    const loading = document.getElementById('models-loading');
    const error = document.getElementById('models-error');
    
    if (!container) return;
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    container.innerHTML = '';
    
    try {
        const response = await WebScoutAPI.makeRequest('/v1/models', 'GET', null, false);
        
        if (response.status === 200 && response.data.data) {
            const models = response.data.data;
            displayModels(models);
            updateModelStats(models);
        } else {
            throw new Error('Failed to load models');
        }
    } catch (err) {
        error.style.display = 'block';
        WebScoutApp.showToast('Failed to load models', 'error');
    } finally {
        loading.style.display = 'none';
    }
}

function displayModels(models) {
    const container = document.getElementById('models-container');
    if (!container) return;
    
    container.innerHTML = models.map(model => `
        <div class="model-card" data-model-type="${model.object}" data-model-name="${model.id.toLowerCase()}">
            <div class="model-header">
                <div>
                    <div class="model-name">${model.id}</div>
                    <div class="model-provider">by ${model.owned_by}</div>
                </div>
                <div class="model-type">${model.object}</div>
            </div>
            <div class="model-description">
                ${getModelDescription(model.id)}
            </div>
            <div class="model-meta">
                <span class="model-created">Created: ${formatTimestamp(model.created * 1000)}</span>
                <span class="model-id">ID: ${model.id}</span>
            </div>
        </div>
    `).join('');
}

function getModelDescription(modelId) {
    const descriptions = {
        'gpt-4': 'Most capable GPT model, great for complex tasks requiring deep understanding.',
        'gpt-3.5-turbo': 'Fast and efficient model, perfect for most conversational tasks.',
        'claude-3-sonnet': 'Anthropic\'s balanced model with strong reasoning capabilities.',
        'dall-e-2': 'AI image generation model that creates images from text descriptions.',
        'dall-e-3': 'Latest image generation model with improved quality and prompt adherence.'
    };
    
    return descriptions[modelId] || 'Advanced AI model for various tasks and applications.';
}

function updateModelStats(models) {
    const totalModels = document.getElementById('total-models');
    const chatModels = document.getElementById('chat-models');
    const imageModels = document.getElementById('image-models');
    const providersCount = document.getElementById('providers-count');
    
    if (totalModels) totalModels.textContent = models.length;
    
    const chatCount = models.filter(m => m.object === 'model' || m.id.includes('gpt') || m.id.includes('claude')).length;
    const imageCount = models.filter(m => m.id.includes('dall-e') || m.id.includes('stable')).length;
    const providers = new Set(models.map(m => m.owned_by)).size;
    
    if (chatModels) chatModels.textContent = chatCount;
    if (imageModels) imageModels.textContent = imageCount;
    if (providersCount) providersCount.textContent = providers;
}

// Initialize API module
document.addEventListener('DOMContentLoaded', () => {
    WebScoutAPI.init();
});
