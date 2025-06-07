/**
 * Modern Theme Toggle System for WebScout
 * Provides smooth theme switching with advanced animations
 */

class ThemeToggle {
    constructor() {
        this.currentTheme = localStorage.getItem('webscout-theme') || 'dark';
        this.init();
    }
    
    init() {
        this.createToggleButton();
        this.applyTheme(this.currentTheme);
        this.bindEvents();
    }
    
    createToggleButton() {
        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'theme-toggle-container';
        toggleContainer.innerHTML = `
            <button class="theme-toggle-btn glass-effect" id="theme-toggle" aria-label="Toggle theme">
                <div class="toggle-icon">
                    <svg class="sun-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"/>
                        <line x1="12" y1="1" x2="12" y2="3"/>
                        <line x1="12" y1="21" x2="12" y2="23"/>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                        <line x1="1" y1="12" x2="3" y2="12"/>
                        <line x1="21" y1="12" x2="23" y2="12"/>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                    </svg>
                    <svg class="moon-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                    </svg>
                </div>
            </button>
        `;
        
        // Add to header
        const header = document.querySelector('.header-content');
        if (header) {
            header.appendChild(toggleContainer);
        }
    }
    
    bindEvents() {
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleTheme());
        }
    }
    
    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
        this.currentTheme = newTheme;
        localStorage.setItem('webscout-theme', newTheme);
        
        // Add animation effect
        this.animateThemeChange();
    }
    
    applyTheme(theme) {
        const root = document.documentElement;
        const toggleBtn = document.getElementById('theme-toggle');
        
        if (theme === 'light') {
            root.style.setProperty('--background-color', '#f8fafc');
            root.style.setProperty('--background-secondary', '#ffffff');
            root.style.setProperty('--surface-color', '#ffffff');
            root.style.setProperty('--surface-light', '#f1f5f9');
            root.style.setProperty('--surface-glass', 'rgba(255, 255, 255, 0.8)');
            root.style.setProperty('--text-primary', '#0f172a');
            root.style.setProperty('--text-secondary', '#475569');
            root.style.setProperty('--text-muted', '#94a3b8');
            root.style.setProperty('--border-color', '#e2e8f0');
            root.style.setProperty('--border-glass', 'rgba(0, 0, 0, 0.1)');
            
            if (toggleBtn) {
                toggleBtn.classList.add('light-mode');
            }
        } else {
            root.style.setProperty('--background-color', '#0a0f1c');
            root.style.setProperty('--background-secondary', '#0f172a');
            root.style.setProperty('--surface-color', '#1e293b');
            root.style.setProperty('--surface-light', '#334155');
            root.style.setProperty('--surface-glass', 'rgba(30, 41, 59, 0.8)');
            root.style.setProperty('--text-primary', '#f8fafc');
            root.style.setProperty('--text-secondary', '#cbd5e1');
            root.style.setProperty('--text-muted', '#64748b');
            root.style.setProperty('--border-color', '#334155');
            root.style.setProperty('--border-glass', 'rgba(255, 255, 255, 0.1)');
            
            if (toggleBtn) {
                toggleBtn.classList.remove('light-mode');
            }
        }
    }
    
    animateThemeChange() {
        const overlay = document.createElement('div');
        overlay.className = 'theme-transition-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, var(--primary-color) 0%, transparent 70%);
            opacity: 0;
            pointer-events: none;
            z-index: 9999;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(overlay);
        
        // Animate overlay
        requestAnimationFrame(() => {
            overlay.style.opacity = '0.3';
            setTimeout(() => {
                overlay.style.opacity = '0';
                setTimeout(() => {
                    document.body.removeChild(overlay);
                }, 300);
            }, 150);
        });
    }
}

// CSS for theme toggle
const themeToggleCSS = `
.theme-toggle-container {
    position: absolute;
    top: var(--spacing-lg);
    right: var(--spacing-lg);
    z-index: 10;
}

.theme-toggle-btn {
    width: 50px;
    height: 50px;
    border: none;
    border-radius: var(--radius-full);
    background: var(--surface-glass);
    backdrop-filter: var(--backdrop-blur);
    border: 1px solid var(--border-glass);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition-bounce);
    box-shadow: var(--shadow-glass);
    position: relative;
    overflow: hidden;
}

.theme-toggle-btn:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: var(--shadow-glow);
}

.toggle-icon {
    position: relative;
    width: 24px;
    height: 24px;
}

.toggle-icon svg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    transition: all var(--transition-normal);
}

.sun-icon {
    opacity: 0;
    transform: rotate(180deg) scale(0.5);
}

.moon-icon {
    opacity: 1;
    transform: rotate(0deg) scale(1);
}

.theme-toggle-btn.light-mode .sun-icon {
    opacity: 1;
    transform: rotate(0deg) scale(1);
}

.theme-toggle-btn.light-mode .moon-icon {
    opacity: 0;
    transform: rotate(-180deg) scale(0.5);
}

@media (max-width: 768px) {
    .theme-toggle-container {
        top: var(--spacing-md);
        right: var(--spacing-md);
    }
    
    .theme-toggle-btn {
        width: 44px;
        height: 44px;
    }
    
    .toggle-icon {
        width: 20px;
        height: 20px;
    }
}
`;

// Inject CSS
const style = document.createElement('style');
style.textContent = themeToggleCSS;
document.head.appendChild(style);

// Initialize theme toggle when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new ThemeToggle();
});
