// Web Audio API - Calming Bangladeshi River & Water Ripple Soundscape
class RiverSoundscape {
    constructor() {
        this.ctx = null;
        this.isPlaying = false;
        this.gainNode = null;
        this.filterNode = null;
        this.lfo = null;
        this.noiseNode = null;
    }

    init() {
        if (this.ctx) return;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContext();

        // Master Gain
        this.gainNode = this.ctx.createGain();
        this.gainNode.gain.setValueAtTime(0.0, this.ctx.currentTime);
        this.gainNode.connect(this.ctx.destination);

        // Low-pass filter for water softness
        this.filterNode = this.ctx.createBiquadFilter();
        this.filterNode.type = 'lowpass';
        this.filterNode.frequency.setValueAtTime(450, this.ctx.currentTime);
        this.filterNode.Q.setValueAtTime(3.0, this.ctx.currentTime);

        // Brown noise generator (gentle natural river murmur)
        const bufferSize = this.ctx.sampleRate * 2;
        const noiseBuffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        let lastOut = 0.0;
        for (let i = 0; i < bufferSize; i++) {
            const white = Math.random() * 2 - 1;
            output[i] = (lastOut + (0.02 * white)) / 1.02;
            lastOut = output[i];
            output[i] *= 3.5; // Gain compensation
        }

        this.noiseNode = this.ctx.createBufferSource();
        this.noiseNode.buffer = noiseBuffer;
        this.noiseNode.loop = true;

        // Gentle LFO modulating water swell (simulates river lap every 4 seconds)
        this.lfo = this.ctx.createOscillator();
        this.lfo.frequency.setValueAtTime(0.25, this.ctx.currentTime); // 0.25 Hz = 4s gentle wave
        const lfoGain = this.ctx.createGain();
        lfoGain.gain.setValueAtTime(150, this.ctx.currentTime);

        this.lfo.connect(lfoGain);
        lfoGain.connect(this.filterNode.frequency);

        // Connect chain: Noise -> Filter -> Gain -> Speakers
        this.noiseNode.connect(this.filterNode);
        this.filterNode.connect(this.gainNode);

        this.noiseNode.start();
        this.lfo.start();
    }

    toggle() {
        if (!this.ctx) {
            this.init();
        }

        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        if (this.isPlaying) {
            // Fade out
            this.gainNode.gain.linearRampToValueAtTime(0.001, this.ctx.currentTime + 1.2);
            setTimeout(() => {
                this.isPlaying = false;
                this.updateUI(false);
            }, 1200);
        } else {
            // Fade in gently to mild ambient volume
            this.gainNode.gain.cancelScheduledValues(this.ctx.currentTime);
            this.gainNode.gain.setValueAtTime(0.001, this.ctx.currentTime);
            this.gainNode.gain.linearRampToValueAtTime(0.12, this.ctx.currentTime + 2.0);
            this.isPlaying = true;
            this.updateUI(true);
        }
    }

    updateUI(active) {
        const btn = document.getElementById('audioToggleBtn');
        const waves = document.getElementById('soundWaveAnimation');
        const text = document.getElementById('audioBtnText');
        if (!btn) return;

        if (active) {
            btn.classList.add('audio-active');
            if (waves) waves.style.display = 'inline-flex';
            if (text) text.setAttribute('data-i18n', 'audio_on');
        } else {
            btn.classList.remove('audio-active');
            if (waves) waves.style.display = 'none';
            if (text) text.setAttribute('data-i18n', 'audio_off');
        }
        
        // Refresh text from current i18n
        if (window.updateLanguageUI) window.updateLanguageUI();
    }
}

window.riverSound = new RiverSoundscape();
