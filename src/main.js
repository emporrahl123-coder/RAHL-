import { RahlAI } from './core/ai-engine/neural-reasoner.js';
import { PWAManager } from './pwa/service-worker/sw-core.js';
import { UIManager } from './ui/components/chat-interface.js';
import { SecurityManager } from './utils/security/auth-manager.js';

class RahlApplication {
  constructor() {
    this.ai = new RahlAI();
    this.pwa = new PWAManager();
    this.ui = new UIManager();
    this.security = new SecurityManager();
    
    this.init();
  }
  
  async init() {
    try {
      // Initialize with priority
      await this.security.initialize();
      await this.pwa.registerServiceWorker();
      await this.ai.loadModels();
      await this.ui.render();
      
      this.startRealTimeProcessing();
      this.setupEventListeners();
      
      console.log('RAHL AI PWA initialized successfully');
    } catch (error) {
      console.error('Failed to initialize RAHL:', error);
    }
  }
  
  startRealTimeProcessing() {
    // Start WebSocket connection
    this.socket = new WebSocket('wss://api.rahl.ai/realtime');
    
    // Start background AI processing
    this.backgroundWorker = new Worker('./pwa/web-workers/ai-worker.js');
    
    // Start sensor processing if available
    if ('DeviceOrientationEvent' in window) {
      this.startSensorProcessing();
    }
  }
  
  async processInput(input, modality = 'text') {
    const result = await this.ai.process({
      input,
      modality,
      context: this.ui.getContext(),
      userPreferences: this.security.getUserPreferences()
    });
    
    // Cache result for offline use
    await this.pwa.cacheResult(input, result);
    
    return result;
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.rahl = new RahlApplication();
});

// Export for module usage
export { RahlApplication };
