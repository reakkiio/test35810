/**
 * WebScout API Documentation - Main JavaScript
 * Handles UI interactions, API testing, and dynamic content
 */

// Global application state
window.WebScoutApp = {
    config: {},
    currentApiKey: '',
    activeTab: 'endpoints',
    
    // Initialize the application
    init() {
        console.log('WebScout App initializing...');
        this.config = window.WEBSCOUT_CONFIG || {};
        this.loadSavedApiKey();
        this.setupEventListeners();
        this.initializePage();
        this.initScrollIndicator();
        this.hideLoadingScreen();
    },
    
    // Load saved API key from localStorage
    loadSavedApiKey() {
        const savedApiKey = localStorage.getItem('webscout_api_key');
        if (savedApiKey) {
            this.currentApiKey = savedApiKey;
        }
    },
    
    // Setup all event listeners
    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.closest('.nav-tab').dataset.tab;
                this.showTab(tabName);
            });
        });

        // Endpoint toggles - add direct event listeners
        this.setupEndpointToggles();
        
        // Example language tabs
        document.querySelectorAll('.example-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const language = e.target.closest('.example-tab').dataset.language;
                this.showExampleLanguage(language);
            });
        });
        
        // View toggle for models
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const view = e.target.closest('.view-btn').dataset.view;
                this.toggleModelsView(view);
            });
        });
        
        // Search functionality
        const searchInput = document.getElementById('model-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchModels(e.target.value);
            });
        }
        
        // Filter functionality
        const filterSelect = document.getElementById('model-filter');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.filterModels(e.target.value);
            });
        }
        
        // Copy code buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-code-btn') || e.target.closest('.copy-code-btn')) {
                const btn = e.target.closest('.copy-code-btn');
                const codeId = btn.getAttribute('onclick')?.match(/copyCode\('([^']+)'\)/)?.[1];
                if (codeId) {
                    this.copyCode(codeId);
                }
            }
        });
    },

    // Setup endpoint toggle functionality
    setupEndpointToggles() {
        // Wait for DOM to be fully ready
        setTimeout(() => {
            const headers = document.querySelectorAll('.endpoint-header');
            console.log('Setting up endpoint toggles for', headers.length, 'headers');

            headers.forEach((header, index) => {
                // Remove any existing listeners
                header.replaceWith(header.cloneNode(true));
                const newHeader = document.querySelectorAll('.endpoint-header')[index];

                newHeader.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log(`Endpoint header ${index} clicked`);
                    this.toggleEndpoint(newHeader);
                });

                // Ensure it's clickable
                newHeader.style.cursor = 'pointer';
                console.log(`Added click handler to header ${index}`);
            });
        }, 100);
    },

    // Initialize page content
    initializePage() {
        // Update status indicator
        this.updateStatusIndicator();
        
        // Initialize syntax highlighting
        if (typeof hljs !== 'undefined') {
            hljs.highlightAll();
        }
        
        // Set up periodic status checks
        setInterval(() => {
            this.updateStatusIndicator();
        }, 30000);
    },
    
    // Hide loading screen
    hideLoadingScreen() {
        setTimeout(() => {
            const loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {
                loadingScreen.classList.add('hidden');
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 300);
            }
        }, 1000);
    },
    
    // Show specific tab
    showTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active class from all nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
            tab.setAttribute('aria-selected', 'false');
        });
        
        // Show selected tab content
        const targetTab = document.getElementById(`${tabName}-panel`);
        if (targetTab) {
            targetTab.classList.add('active');
        }
        
        // Add active class to selected nav tab
        const activeNavTab = document.getElementById(`${tabName}-tab`);
        if (activeNavTab) {
            activeNavTab.classList.add('active');
            activeNavTab.setAttribute('aria-selected', 'true');
        }
        
        this.activeTab = tabName;
        
        // Load models if models tab is selected
        if (tabName === 'models') {
            this.loadModels();
        }
    },

    // Load models for the models tab
    async loadModels() {
        if (typeof loadModels === 'function') {
            await loadModels();
        }
    },

    // Search models
    searchModels(query) {
        if (typeof WebScoutUI !== 'undefined' && WebScoutUI.searchModels) {
            WebScoutUI.searchModels(query);
        }
    },

    // Filter models
    filterModels(filter) {
        if (typeof WebScoutUI !== 'undefined' && WebScoutUI.filterModels) {
            WebScoutUI.filterModels(filter);
        }
    },
    
    // Toggle endpoint expansion
    toggleEndpoint(header) {
        console.log('toggleEndpoint called with:', header);
        const body = header.nextElementSibling;
        const icon = header.querySelector('.expand-icon');

        console.log('Body element:', body);
        console.log('Icon element:', icon);

        if (body && icon) {
            const isExpanded = header.classList.contains('expanded');
            console.log('Is expanded:', isExpanded);

            if (isExpanded) {
                header.classList.remove('expanded');
                body.classList.remove('expanded');
                body.style.display = 'none';
                console.log('Collapsed endpoint');
            } else {
                header.classList.add('expanded');
                body.classList.add('expanded');
                body.style.display = 'block';
                console.log('Expanded endpoint');
            }

            // Add animation class for smooth transition
            body.classList.add('animate-slideDown');
            setTimeout(() => {
                body.classList.remove('animate-slideDown');
            }, 300);
        } else {
            console.log('Missing body or icon element');
        }
    },
    
    // Show example language
    showExampleLanguage(language) {
        // Hide all language examples
        document.querySelectorAll('.language-examples').forEach(example => {
            example.classList.remove('active');
        });
        
        // Remove active class from all example tabs
        document.querySelectorAll('.example-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected language examples
        const targetExample = document.getElementById(`${language}-examples`);
        if (targetExample) {
            targetExample.classList.add('active');
        }
        
        // Add active class to selected example tab
        const activeTab = document.querySelector(`[data-language="${language}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
        
        // Re-highlight syntax
        if (typeof hljs !== 'undefined') {
            setTimeout(() => {
                hljs.highlightAll();
            }, 100);
        }
    },
    
    // Toggle models view (grid/list)
    toggleModelsView(view) {
        const container = document.getElementById('models-container');
        const buttons = document.querySelectorAll('.view-btn');
        
        if (container) {
            container.className = view === 'list' ? 'models-list' : 'models-grid';
        }
        
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });
    },
    
    // Copy code to clipboard
    copyCode(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            const text = element.textContent || element.innerText;
            navigator.clipboard.writeText(text).then(() => {
                this.showToast('Code copied to clipboard!', 'success');
            }).catch(() => {
                this.showToast('Failed to copy code', 'error');
            });
        }
    },
    
    // Update status indicator
    async updateStatusIndicator() {
        try {
            const response = await fetch(`${this.config.baseUrl}/health`);
            const indicator = document.getElementById('status-indicator');
            if (indicator) {
                indicator.textContent = response.ok ? '🟢' : '🔴';
            }
        } catch {
            const indicator = document.getElementById('status-indicator');
            if (indicator) {
                indicator.textContent = '🔴';
            }
        }
    },
    
    // Initialize scroll indicator
    initScrollIndicator() {
        const scrollProgress = document.getElementById('scroll-progress');
        if (!scrollProgress) return;

        const updateScrollProgress = () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = (scrollTop / scrollHeight) * 100;
            scrollProgress.style.width = `${Math.min(progress, 100)}%`;
        };

        window.addEventListener('scroll', updateScrollProgress);
        updateScrollProgress(); // Initial call
    },

    // Toggle advanced parameters
    toggleAdvanced(section) {
        const advancedParams = document.getElementById(`${section}-advanced`);
        const toggleBtn = document.querySelector(`[onclick="toggleAdvanced('${section}')"]`);

        if (advancedParams && toggleBtn) {
            const isVisible = advancedParams.classList.contains('show');

            if (isVisible) {
                advancedParams.classList.remove('show');
                toggleBtn.classList.remove('expanded');
            } else {
                advancedParams.classList.add('show');
                toggleBtn.classList.add('expanded');
            }
        }
    },

    // Test chat completion endpoint with all parameters
    async testChatCompletion(event) {
        event.preventDefault();

        const startTime = Date.now();

        // Get all form values
        const model = document.getElementById('chat-model').value;
        const messages = document.getElementById('chat-messages').value;
        const temperature = parseFloat(document.getElementById('chat-temperature').value);
        const maxTokens = parseInt(document.getElementById('chat-max-tokens').value);
        const topP = parseFloat(document.getElementById('chat-top-p').value);
        const frequencyPenalty = parseFloat(document.getElementById('chat-frequency-penalty').value);
        const presencePenalty = parseFloat(document.getElementById('chat-presence-penalty').value);
        const n = parseInt(document.getElementById('chat-n').value);
        const stream = document.getElementById('chat-stream').checked;
        const echo = document.getElementById('chat-echo').checked;

        // Advanced parameters
        const stop = document.getElementById('chat-stop').value;
        const user = document.getElementById('chat-user').value;
        const seed = document.getElementById('chat-seed').value;
        const logitBias = document.getElementById('chat-logit-bias').value;

        if (!messages.trim()) {
            this.showToast('Please enter messages', 'error');
            return;
        }

        let parsedMessages;
        try {
            parsedMessages = JSON.parse(messages);
        } catch (error) {
            this.showToast('Invalid JSON format for messages', 'error');
            return;
        }

        // Build request data
        const requestData = {
            model: model,
            messages: parsedMessages,
            temperature: temperature,
            max_tokens: maxTokens,
            top_p: topP,
            frequency_penalty: frequencyPenalty,
            presence_penalty: presencePenalty,
            n: n,
            stream: stream
        };

        // Add optional parameters if provided
        if (echo) requestData.echo = echo;
        if (stop && stop.trim()) {
            try {
                requestData.stop = JSON.parse(stop);
            } catch (error) {
                requestData.stop = stop.split(',').map(s => s.trim());
            }
        }
        if (user && user.trim()) requestData.user = user;
        if (seed && seed.trim()) requestData.seed = parseInt(seed);
        if (logitBias && logitBias.trim()) {
            try {
                requestData.logit_bias = JSON.parse(logitBias);
            } catch (error) {
                this.showToast('Invalid JSON format for logit bias', 'error');
                return;
            }
        }

        const responseDiv = document.getElementById('chat-response');
        responseDiv.innerHTML = `
            <div class="loading modern-loading">
                <div class="loading-spinner"></div>
                <span>Sending request...</span>
            </div>
        `;

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer your-api-key'
                },
                body: JSON.stringify(requestData)
            });

            const responseTime = Date.now() - startTime;

            // Handle streaming response
            if (stream && response.body) {
                responseDiv.innerHTML = `
                    <div class="response-header modern-response-header">
                        <div class="response-status">
                            <span class="status-badge status-${response.status}">${response.status} ${response.statusText}</span>
                            <span class="response-time">Streaming...</span>
                        </div>
                        <button class="copy-btn" onclick="copyStreamContent()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                            Copy
                        </button>
                    </div>
                    <div class="response-body modern-response-body streaming">
                        <div class="stream-content" id="stream-content"></div>
                        <div class="stream-raw" id="stream-raw" style="display: none;"></div>
                        <div class="stream-controls">
                            <button class="stream-toggle" onclick="toggleStreamView()">Show Raw</button>
                        </div>
                    </div>
                `;

                await this.handleStreamingResponse(response, startTime);
            } else {
                // Handle regular response
                const result = await response.json();

                responseDiv.innerHTML = `
                    <div class="response-header modern-response-header">
                        <div class="response-status">
                            <span class="status-badge status-${response.status}">${response.status} ${response.statusText}</span>
                            <span class="response-time">${responseTime}ms</span>
                        </div>
                        <button class="copy-btn" onclick="copyToClipboard('${JSON.stringify(result).replace(/'/g, "\\'")}')">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                            Copy
                        </button>
                    </div>
                    <pre class="response-body modern-response-body"><code class="language-json">${JSON.stringify(result, null, 2)}</code></pre>
                `;
            }

            this.showToast('Request completed successfully', 'success');
        } catch (error) {
            const responseTime = Date.now() - startTime;
            responseDiv.innerHTML = `
                <div class="response-header modern-response-header error">
                    <div class="response-status">
                        <span class="status-badge status-error">Error</span>
                        <span class="response-time">${responseTime}ms</span>
                    </div>
                </div>
                <div class="response-body modern-response-body error">
                    <div class="error-content">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="error-icon">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="15" y1="9" x2="9" y2="15"/>
                            <line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                        <div>
                            <strong>Request Failed</strong>
                            <p>${error.message}</p>
                        </div>
                    </div>
                </div>
            `;
            this.showToast('Request failed', 'error');
        }
    },

    // Handle streaming response
    async handleStreamingResponse(response, startTime) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const streamContent = document.getElementById('stream-content');
        const streamRaw = document.getElementById('stream-raw');

        let fullContent = '';
        let rawData = '';
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    const finalTime = Date.now() - startTime;
                    document.querySelector('.response-time').textContent = `${finalTime}ms`;
                    break;
                }

                // Decode the chunk
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                rawData += chunk;

                // Process complete lines
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.trim() === '') continue;

                    if (line.startsWith('data: ')) {
                        const data = line.slice(6); // Remove 'data: ' prefix

                        if (data === '[DONE]') {
                            streamContent.innerHTML += '<div class="stream-done">✅ Stream completed</div>';
                            continue;
                        }

                        try {
                            const parsed = JSON.parse(data);

                            // Extract content from delta
                            if (parsed.choices && parsed.choices[0] && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                                const content = parsed.choices[0].delta.content;
                                fullContent += content;

                                // Update the displayed content
                                streamContent.innerHTML = `
                                    <div class="stream-message">
                                        <strong>Generated Response:</strong>
                                        <div class="stream-text">${fullContent}</div>
                                    </div>
                                `;
                            }

                            // Add to raw display
                            streamRaw.innerHTML += `<div class="stream-chunk">data: ${JSON.stringify(parsed, null, 2)}</div>`;

                        } catch (e) {
                            // Handle non-JSON data lines
                            streamRaw.innerHTML += `<div class="stream-chunk">data: ${data}</div>`;
                        }
                    }
                }

                // Auto-scroll to bottom
                streamContent.scrollTop = streamContent.scrollHeight;
                streamRaw.scrollTop = streamRaw.scrollHeight;
            }
        } catch (error) {
            streamContent.innerHTML += `<div class="stream-error">❌ Stream error: ${error.message}</div>`;
        }

        // Store full content for copying
        window.streamFullContent = fullContent;
        window.streamRawData = rawData;
    },

    // Test image generation endpoint
    async testImageGeneration(event) {
        event.preventDefault();

        const startTime = Date.now();

        // Get form values
        const model = document.getElementById('image-model').value;
        const prompt = document.getElementById('image-prompt').value;
        const size = document.getElementById('image-size').value;
        const count = parseInt(document.getElementById('image-count').value);

        if (!prompt.trim()) {
            this.showToast('Please enter a prompt', 'error');
            return;
        }

        // Build request data
        const requestData = {
            model: model,
            prompt: prompt,
            size: size,
            n: count
        };

        const responseDiv = document.getElementById('image-response');
        responseDiv.innerHTML = `
            <div class="loading modern-loading">
                <div class="loading-spinner"></div>
                <span>Generating image...</span>
            </div>
        `;

        try {
            const response = await fetch('/v1/images/generations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer your-api-key'
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            const responseTime = Date.now() - startTime;

            responseDiv.innerHTML = `
                <div class="response-header modern-response-header">
                    <div class="response-status">
                        <span class="status-badge status-${response.status}">${response.status} ${response.statusText}</span>
                        <span class="response-time">${responseTime}ms</span>
                    </div>
                    <button class="copy-btn" onclick="copyToClipboard('${JSON.stringify(result).replace(/'/g, "\\'")}')">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                        Copy
                    </button>
                </div>
                <pre class="response-body modern-response-body"><code class="language-json">${JSON.stringify(result, null, 2)}</code></pre>
            `;

            this.showToast('Image generation completed successfully', 'success');
        } catch (error) {
            const responseTime = Date.now() - startTime;
            responseDiv.innerHTML = `
                <div class="response-header modern-response-header error">
                    <div class="response-status">
                        <span class="status-badge status-error">Error</span>
                        <span class="response-time">${responseTime}ms</span>
                    </div>
                </div>
                <div class="response-body modern-response-body error">
                    <div class="error-content">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="error-icon">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="15" y1="9" x2="9" y2="15"/>
                            <line x1="9" y1="9" x2="15" y2="15"/>
                        </svg>
                        <div>
                            <strong>Image Generation Failed</strong>
                            <p>${error.message}</p>
                        </div>
                    </div>
                </div>
            `;
            this.showToast('Image generation failed', 'error');
        }
    },

    // Show toast notification
    showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type} animate-slideInRight`;

        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <div class="toast-content">
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">×</button>
        `;

        container.appendChild(toast);

        // Auto remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('animate-slideOutRight');
                setTimeout(() => {
                    if (toast.parentElement) {
                        toast.remove();
                    }
                }, 300);
            }
        }, duration);
    }
};

// Global functions for backward compatibility
function showTab(tabName) {
    WebScoutApp.showTab(tabName);
}

function toggleEndpoint(header) {
    WebScoutApp.toggleEndpoint(header);
}

function copyCode(elementId) {
    WebScoutApp.copyCode(elementId);
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        WebScoutApp.showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        WebScoutApp.showToast('Failed to copy', 'error');
    });
}

// Global function for advanced toggle (called from HTML)
function toggleAdvanced(section) {
    WebScoutApp.toggleAdvanced(section);
}

// Global function for testing chat completion (called from HTML)
function testChatCompletion(event) {
    WebScoutApp.testChatCompletion(event);
}

// Global function for testing image generation (called from HTML)
function testImageGeneration(event) {
    WebScoutApp.testImageGeneration(event);
}

// Global function to toggle stream view
function toggleStreamView() {
    const streamContent = document.getElementById('stream-content');
    const streamRaw = document.getElementById('stream-raw');
    const toggleBtn = document.querySelector('.stream-toggle');

    if (streamRaw.style.display === 'none') {
        streamContent.style.display = 'none';
        streamRaw.style.display = 'block';
        toggleBtn.textContent = 'Show Formatted';
    } else {
        streamContent.style.display = 'block';
        streamRaw.style.display = 'none';
        toggleBtn.textContent = 'Show Raw';
    }
}

// Global function to copy stream content
function copyStreamContent() {
    const content = window.streamFullContent || '';
    if (content) {
        navigator.clipboard.writeText(content).then(() => {
            WebScoutApp.showToast('Stream content copied to clipboard!', 'success');
        }).catch(() => {
            WebScoutApp.showToast('Failed to copy stream content', 'error');
        });
    } else {
        WebScoutApp.showToast('No stream content to copy', 'warning');
    }
}

// Utility functions
function formatJSON(obj) {
    return JSON.stringify(obj, null, 2);
}

function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
}

function generateId() {
    return Math.random().toString(36).substr(2, 9);
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    WebScoutApp.init();
});
