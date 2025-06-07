/**
 * WebScout API Documentation - UI Utilities
 * Handles UI interactions, animations, and helper functions
 */

// UI utility functions
window.WebScoutUI = {
    // Show loading state on button
    showLoading(button, loadingText = 'Loading...') {
        if (!button) return '';
        
        const originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <span>${loadingText}</span>
            </div>
        `;
        return originalText;
    },
    
    // Hide loading state on button
    hideLoading(button, originalText) {
        if (!button || !originalText) return;
        
        button.disabled = false;
        button.innerHTML = originalText;
    },
    
    // Display API response
    displayResponse(containerId, response) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        container.classList.add('show');
        
        // Create response header
        const header = document.createElement('div');
        header.className = 'response-header';
        
        const statusClass = this.getStatusClass(response.status);
        header.innerHTML = `
            <div class="response-status">
                <span class="status-code ${statusClass}">${response.status}</span>
                <span class="status-text">${response.statusText}</span>
            </div>
            <div class="response-time">${response.responseTime}ms</div>
        `;
        
        container.appendChild(header);
        
        // Create response body
        const body = document.createElement('div');
        body.className = 'response-body';
        
        const pre = document.createElement('pre');
        pre.innerHTML = this.formatResponseData(response.data);
        body.appendChild(pre);
        
        container.appendChild(body);
    },
    
    // Display image generation response
    displayImageResponse(containerId, response) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        container.classList.add('show');
        
        // Create response header
        const header = document.createElement('div');
        header.className = 'response-header';
        
        const statusClass = this.getStatusClass(response.status);
        header.innerHTML = `
            <div class="response-status">
                <span class="status-code ${statusClass}">${response.status}</span>
                <span class="status-text">${response.statusText}</span>
            </div>
            <div class="response-time">${response.responseTime}ms</div>
        `;
        
        container.appendChild(header);
        
        // Create response body
        const body = document.createElement('div');
        body.className = 'response-body';
        
        if (response.status === 200 && response.data.data) {
            // Display images
            const imagesContainer = document.createElement('div');
            imagesContainer.className = 'images-container';
            imagesContainer.style.cssText = `
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 1rem;
            `;
            
            response.data.data.forEach((image, index) => {
                const imageCard = document.createElement('div');
                imageCard.className = 'image-card';
                imageCard.style.cssText = `
                    border: 1px solid var(--border-color);
                    border-radius: var(--radius);
                    overflow: hidden;
                    background: var(--surface-color);
                `;
                
                imageCard.innerHTML = `
                    <img src="${image.url}" alt="Generated image ${index + 1}" 
                         style="width: 100%; height: auto; display: block;">
                    <div style="padding: 0.5rem; text-align: center;">
                        <a href="${image.url}" target="_blank" 
                           style="color: var(--primary-color); text-decoration: none; font-size: 0.875rem;">
                            🔗 Open Full Size
                        </a>
                    </div>
                `;
                
                imagesContainer.appendChild(imageCard);
            });
            
            body.appendChild(imagesContainer);
        }
        
        // Add JSON response
        const pre = document.createElement('pre');
        pre.innerHTML = this.formatResponseData(response.data);
        body.appendChild(pre);
        
        container.appendChild(body);
    },
    
    // Get CSS class for status code
    getStatusClass(status) {
        if (status >= 200 && status < 300) return 'status-200';
        if (status >= 400 && status < 500) return 'status-400';
        if (status >= 500) return 'status-500';
        return 'status-400';
    },
    
    // Format response data with syntax highlighting
    formatResponseData(data) {
        if (typeof data === 'string') {
            try {
                data = JSON.parse(data);
            } catch (e) {
                return this.escapeHtml(data);
            }
        }
        
        const jsonString = JSON.stringify(data, null, 2);
        return this.highlightJSON(jsonString);
    },
    
    // Simple JSON syntax highlighting
    highlightJSON(json) {
        return json
            .replace(/(".*?")\s*:/g, '<span class="json-key">$1</span>:')
            .replace(/:\s*(".*?")/g, ': <span class="json-string">$1</span>')
            .replace(/:\s*(\d+\.?\d*)/g, ': <span class="json-number">$1</span>')
            .replace(/:\s*(true|false)/g, ': <span class="json-boolean">$1</span>')
            .replace(/:\s*(null)/g, ': <span class="json-null">$1</span>');
    },
    
    // Escape HTML
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },
    
    // Search models
    searchModels(query) {
        const cards = document.querySelectorAll('.model-card');
        const noResults = document.getElementById('no-models');
        let visibleCount = 0;
        
        cards.forEach(card => {
            const modelName = card.dataset.modelName || '';
            const isVisible = modelName.includes(query.toLowerCase());
            card.style.display = isVisible ? 'block' : 'none';
            if (isVisible) visibleCount++;
        });
        
        if (noResults) {
            noResults.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    },
    
    // Filter models
    filterModels(filter) {
        const cards = document.querySelectorAll('.model-card');
        const noResults = document.getElementById('no-models');
        let visibleCount = 0;
        
        cards.forEach(card => {
            const modelType = card.dataset.modelType || '';
            const modelName = card.dataset.modelName || '';
            
            let isVisible = filter === 'all';
            
            if (filter === 'chat') {
                isVisible = modelType === 'model' || modelName.includes('gpt') || modelName.includes('claude');
            } else if (filter === 'image') {
                isVisible = modelName.includes('dall-e') || modelName.includes('stable');
            } else if (filter === 'embedding') {
                isVisible = modelName.includes('embedding') || modelName.includes('ada');
            }
            
            card.style.display = isVisible ? 'block' : 'none';
            if (isVisible) visibleCount++;
        });
        
        if (noResults) {
            noResults.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    },
    
    // Download Postman collection
    downloadPostmanCollection() {
        const collection = {
            info: {
                name: "WebScout API",
                description: "WebScout OpenAI-compatible API collection",
                schema: "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            item: [
                {
                    name: "Chat Completions",
                    request: {
                        method: "POST",
                        header: [
                            {
                                key: "Authorization",
                                value: "Bearer {{api_key}}",
                                type: "text"
                            },
                            {
                                key: "Content-Type",
                                value: "application/json",
                                type: "text"
                            }
                        ],
                        body: {
                            mode: "raw",
                            raw: JSON.stringify({
                                model: "gpt-3.5-turbo",
                                messages: [
                                    {
                                        role: "user",
                                        content: "Hello, how are you?"
                                    }
                                ],
                                temperature: 0.7,
                                max_tokens: 150
                            }, null, 2)
                        },
                        url: {
                            raw: "{{base_url}}/v1/chat/completions",
                            host: ["{{base_url}}"],
                            path: ["v1", "chat", "completions"]
                        }
                    }
                },
                {
                    name: "List Models",
                    request: {
                        method: "GET",
                        header: [
                            {
                                key: "Authorization",
                                value: "Bearer {{api_key}}",
                                type: "text"
                            }
                        ],
                        url: {
                            raw: "{{base_url}}/v1/models",
                            host: ["{{base_url}}"],
                            path: ["v1", "models"]
                        }
                    }
                },
                {
                    name: "Generate Image",
                    request: {
                        method: "POST",
                        header: [
                            {
                                key: "Authorization",
                                value: "Bearer {{api_key}}",
                                type: "text"
                            },
                            {
                                key: "Content-Type",
                                value: "application/json",
                                type: "text"
                            }
                        ],
                        body: {
                            mode: "raw",
                            raw: JSON.stringify({
                                prompt: "A beautiful sunset over mountains",
                                model: "dall-e-2",
                                size: "512x512",
                                n: 1
                            }, null, 2)
                        },
                        url: {
                            raw: "{{base_url}}/v1/images/generations",
                            host: ["{{base_url}}"],
                            path: ["v1", "images", "generations"]
                        }
                    }
                }
            ],
            variable: [
                {
                    key: "base_url",
                    value: window.WEBSCOUT_CONFIG?.baseUrl || "http://localhost:8000"
                },
                {
                    key: "api_key",
                    value: "your-api-key-here"
                }
            ]
        };
        
        const blob = new Blob([JSON.stringify(collection, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'webscout-api.postman_collection.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        WebScoutApp.showToast('Postman collection downloaded!', 'success');
    }
};

// Global functions for backward compatibility
function showLoading(button, loadingText) {
    return WebScoutUI.showLoading(button, loadingText);
}

function hideLoading(button, originalText) {
    WebScoutUI.hideLoading(button, originalText);
}

function displayResponse(containerId, response) {
    WebScoutUI.displayResponse(containerId, response);
}

function displayImageResponse(containerId, response) {
    WebScoutUI.displayImageResponse(containerId, response);
}

function searchModels(query) {
    WebScoutUI.searchModels(query);
}

function filterModels(filter) {
    WebScoutUI.filterModels(filter);
}

function downloadPostmanCollection() {
    WebScoutUI.downloadPostmanCollection();
}
