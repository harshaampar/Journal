# Entrainment Methods Research Notes

## Binaural and Isochronic Tones Overview

### **Binaural Beats**
- **How they work**: Two slightly different frequencies played in each ear (e.g., 440Hz left, 446Hz right)
- **Brain perception**: Creates a perceived "beat" at the difference frequency (6Hz in this example)
- **Requirements**: Stereo headphones essential - won't work through speakers
- **Mechanism**: Brain synchronizes to the beat frequency through frequency following response

### **Isochronic Tones**
- **How they work**: Single tone that pulses on/off at regular intervals
- **Brain perception**: Direct rhythmic stimulation at the pulse rate
- **Requirements**: Works through speakers or headphones
- **Mechanism**: More direct entrainment through amplitude modulation

### **Monaural Beats**
- **How they work**: Two frequencies mixed together before reaching the ears (e.g., 440Hz + 446Hz combined)
- **Brain perception**: Physical beat frequency created by acoustic interference (6Hz in this example)
- **Requirements**: Works through speakers or headphones, no stereo separation needed
- **Mechanism**: Real acoustic beating creates periodic amplitude variations that drive entrainment

## Brain Entrainment Frequency Ranges

| **Brainwave** | **Frequency** | **Associated States** |
|---------------|---------------|-----------------------|
| Delta (δ) | 0.5-4 Hz | Deep sleep, healing, regeneration |
| Theta (θ) | 4-8 Hz | Deep meditation, creativity, REM sleep |
| Alpha (α) | 8-13 Hz | Relaxed awareness, light meditation |
| Beta (β) | 13-30 Hz | Normal waking consciousness, focus |
| Gamma (γ) | 30-100 Hz | High-level cognitive processing |

## Converting Existing Audio for Entrainment

### **For Binaural Beats:**
```
1. Choose target frequency (e.g., 10Hz for alpha)
2. Select carrier frequency (usually 200-400Hz)
3. Left channel: carrier frequency (e.g., 300Hz)
4. Right channel: carrier + target (e.g., 310Hz for 10Hz beat)
5. Mix with original audio at 10-30% volume
```

### **For Isochronic Tones:**
```
1. Generate sine wave at chosen frequency
2. Apply amplitude modulation at target rate
3. Create on/off pulses (50% duty cycle typical)
4. Layer with original audio
```

### **For Monaural Beats:**
```
1. Choose target frequency (e.g., 10Hz for alpha)
2. Select carrier frequency (usually 200-400Hz)
3. Generate two sine waves: carrier (e.g., 300Hz) and carrier + target (310Hz)
4. Mix both frequencies together into single channel
5. Apply to both left and right channels equally
6. Mix with original audio at 10-30% volume
```

## Technical Implementation Considerations

**Audio Processing Pipeline:**
1. **Frequency selection** - Match desired brainwave state
2. **Volume balancing** - Entrainment tones should be subtle (10-30% of main audio)
3. **Fade in/out** - Gradual introduction prevents jarring
4. **Quality preservation** - Maintain original audio fidelity

**Key Parameters:**
- **Carrier frequency**: 200-500Hz range works best
- **Beat frequency**: Match target brainwave (1-40Hz)
- **Modulation depth**: 50-100% for isochronic
- **Mix ratio**: 70% original, 30% entrainment audio

## Implementation Methods & Pseudocode

### **Binaural Beat Generation**

```javascript
function generateBinauralBeat(originalAudio, targetFrequency, carrierFrequency = 300) {
    const sampleRate = originalAudio.sampleRate;
    const duration = originalAudio.duration;
    const numSamples = sampleRate * duration;
    
    // Generate carrier tones
    const leftCarrier = generateSineWave(carrierFrequency, duration, sampleRate);
    const rightCarrier = generateSineWave(carrierFrequency + targetFrequency, duration, sampleRate);
    
    // Mix with original audio
    const leftChannel = mixAudio(originalAudio.left, leftCarrier, 0.7, 0.3);
    const rightChannel = mixAudio(originalAudio.right, rightCarrier, 0.7, 0.3);
    
    return { left: leftChannel, right: rightChannel };
}

function generateSineWave(frequency, duration, sampleRate) {
    const numSamples = sampleRate * duration;
    const samples = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        samples[i] = Math.sin(2 * Math.PI * frequency * time);
    }
    
    return samples;
}

function mixAudio(audio1, audio2, gain1, gain2) {
    const mixed = new Float32Array(audio1.length);
    
    for (let i = 0; i < audio1.length; i++) {
        mixed[i] = (audio1[i] * gain1) + (audio2[i] * gain2);
        
        // Prevent clipping
        mixed[i] = Math.max(-1, Math.min(1, mixed[i]));
    }
    
    return mixed;
}
```

### **Isochronic Tone Generation**

```javascript
function generateIsochronicTone(originalAudio, targetFrequency, carrierFrequency = 300) {
    const sampleRate = originalAudio.sampleRate;
    const duration = originalAudio.duration;
    const numSamples = sampleRate * duration;
    
    // Generate base carrier wave
    const carrier = generateSineWave(carrierFrequency, duration, sampleRate);
    
    // Apply amplitude modulation at target frequency
    const modulatedCarrier = applyAmplitudeModulation(carrier, targetFrequency, sampleRate);
    
    // Mix with original audio (mono or stereo)
    const leftChannel = mixAudio(originalAudio.left, modulatedCarrier, 0.7, 0.3);
    const rightChannel = mixAudio(originalAudio.right, modulatedCarrier, 0.7, 0.3);
    
    return { left: leftChannel, right: rightChannel };
}

function applyAmplitudeModulation(carrier, modulationFreq, sampleRate, depth = 1.0) {
    const modulated = new Float32Array(carrier.length);
    
    for (let i = 0; i < carrier.length; i++) {
        const time = i / sampleRate;
        
        // Generate square wave modulation (50% duty cycle)
        const modulation = Math.sign(Math.sin(2 * Math.PI * modulationFreq * time));
        
        // Apply modulation (0 for off, 1 for on)
        const amplitude = modulation > 0 ? 1.0 : 0.0;
        
        modulated[i] = carrier[i] * amplitude * depth;
    }
    
    return modulated;
}

// Alternative: Smooth sine wave modulation instead of square wave
function applySmoothModulation(carrier, modulationFreq, sampleRate, depth = 0.5) {
    const modulated = new Float32Array(carrier.length);
    
    for (let i = 0; i < carrier.length; i++) {
        const time = i / sampleRate;
        
        // Sine wave modulation for smoother transitions
        const modulation = Math.sin(2 * Math.PI * modulationFreq * time);
        const amplitude = 0.5 + (depth * modulation * 0.5); // Range: 0.5±(depth*0.5)
        
        modulated[i] = carrier[i] * amplitude;
    }
    
    return modulated;
}
```

### **Monaural Beat Generation**

```javascript
function generateMonauralBeat(originalAudio, targetFrequency, carrierFrequency = 300) {
    const sampleRate = originalAudio.sampleRate;
    const duration = originalAudio.duration;
    const numSamples = sampleRate * duration;
    
    // Generate two carrier frequencies and mix them
    const carrier1 = generateSineWave(carrierFrequency, duration, sampleRate);
    const carrier2 = generateSineWave(carrierFrequency + targetFrequency, duration, sampleRate);
    
    // Mix the two carriers together to create acoustic beating
    const monauralBeat = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
        monauralBeat[i] = (carrier1[i] + carrier2[i]) * 0.5; // Average the two carriers
    }
    
    // Apply the same monaural beat to both channels
    const leftChannel = mixAudio(originalAudio.left, monauralBeat, 0.7, 0.3);
    const rightChannel = mixAudio(originalAudio.right, monauralBeat, 0.7, 0.3);
    
    return { left: leftChannel, right: rightChannel };
}

// Alternative implementation with direct beat frequency calculation
function generateMonauralBeatDirect(originalAudio, targetFrequency, carrierFrequency = 300) {
    const sampleRate = originalAudio.sampleRate;
    const duration = originalAudio.duration;
    const numSamples = sampleRate * duration;
    
    const monauralBeat = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        
        // Generate beating pattern: carrier modulated by beat frequency
        const carrier = Math.sin(2 * Math.PI * carrierFrequency * time);
        const beatEnvelope = Math.cos(2 * Math.PI * targetFrequency * time);
        
        // Create amplitude modulated signal
        monauralBeat[i] = carrier * (0.5 + 0.5 * beatEnvelope);
    }
    
    // Apply to both channels
    const leftChannel = mixAudio(originalAudio.left, monauralBeat, 0.7, 0.3);
    const rightChannel = mixAudio(originalAudio.right, monauralBeat, 0.7, 0.3);
    
    return { left: leftChannel, right: rightChannel };
}
```

### **Advanced: Amplitude Modulation of Existing Audio**

```javascript
function modulateExistingAudio(originalAudio, targetFrequency, depth = 0.3) {
    const sampleRate = originalAudio.sampleRate;
    const numSamples = originalAudio.left.length;
    
    const modulatedLeft = new Float32Array(numSamples);
    const modulatedRight = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        
        // Generate modulation envelope
        const modulation = Math.sin(2 * Math.PI * targetFrequency * time);
        const amplitude = 1.0 + (depth * modulation);
        
        // Apply modulation to original audio
        modulatedLeft[i] = originalAudio.left[i] * amplitude;
        modulatedRight[i] = originalAudio.right[i] * amplitude;
    }
    
    return { left: modulatedLeft, right: modulatedRight };
}
```

### **Frequency Ramping for Progressive Entrainment**

```javascript
function generateRampingBinaural(originalAudio, startFreq, endFreq, rampDuration) {
    const sampleRate = originalAudio.sampleRate;
    const totalDuration = originalAudio.duration;
    const numSamples = sampleRate * totalDuration;
    
    const leftChannel = new Float32Array(numSamples);
    const rightChannel = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        
        // Calculate current target frequency (linear ramp)
        let currentTargetFreq;
        if (time < rampDuration) {
            const progress = time / rampDuration;
            currentTargetFreq = startFreq + (endFreq - startFreq) * progress;
        } else {
            currentTargetFreq = endFreq;
        }
        
        // Generate binaural beat at current frequency
        const carrierFreq = 300;
        const leftTone = Math.sin(2 * Math.PI * carrierFreq * time);
        const rightTone = Math.sin(2 * Math.PI * (carrierFreq + currentTargetFreq) * time);
        
        // Mix with original audio
        leftChannel[i] = (originalAudio.left[i] * 0.7) + (leftTone * 0.3);
        rightChannel[i] = (originalAudio.right[i] * 0.7) + (rightTone * 0.3);
    }
    
    return { left: leftChannel, right: rightChannel };
}
```

### **Real-time Processing Framework**

```javascript
class EntrainmentProcessor {
    constructor(sampleRate, bufferSize = 1024) {
        this.sampleRate = sampleRate;
        this.bufferSize = bufferSize;
        this.phase = 0;
        this.targetFrequency = 10; // Default 10Hz alpha
        this.carrierFrequency = 300;
        this.method = 'binaural'; // 'binaural', 'isochronic', or 'monaural'
    }
    
    processBuffer(inputBuffer) {
        const outputBuffer = {
            left: new Float32Array(this.bufferSize),
            right: new Float32Array(this.bufferSize)
        };
        
        for (let i = 0; i < this.bufferSize; i++) {
            const time = this.phase / this.sampleRate;
            
            if (this.method === 'binaural') {
                const leftTone = Math.sin(2 * Math.PI * this.carrierFrequency * time);
                const rightTone = Math.sin(2 * Math.PI * (this.carrierFrequency + this.targetFrequency) * time);
                
                outputBuffer.left[i] = (inputBuffer.left[i] * 0.7) + (leftTone * 0.3);
                outputBuffer.right[i] = (inputBuffer.right[i] * 0.7) + (rightTone * 0.3);
            } else if (this.method === 'isochronic') {
                const modulation = Math.sign(Math.sin(2 * Math.PI * this.targetFrequency * time));
                const amplitude = modulation > 0 ? 1.0 : 0.0;
                const tone = Math.sin(2 * Math.PI * this.carrierFrequency * time) * amplitude;
                
                outputBuffer.left[i] = (inputBuffer.left[i] * 0.7) + (tone * 0.3);
                outputBuffer.right[i] = (inputBuffer.right[i] * 0.7) + (tone * 0.3);
            } else if (this.method === 'monaural') {
                // Generate monaural beat using amplitude modulation
                const carrier = Math.sin(2 * Math.PI * this.carrierFrequency * time);
                const beatEnvelope = Math.cos(2 * Math.PI * this.targetFrequency * time);
                const monauralTone = carrier * (0.5 + 0.5 * beatEnvelope);
                
                outputBuffer.left[i] = (inputBuffer.left[i] * 0.7) + (monauralTone * 0.3);
                outputBuffer.right[i] = (inputBuffer.right[i] * 0.7) + (monauralTone * 0.3);
            }
            
            this.phase++;
        }
        
        return outputBuffer;
    }
    
    setTargetFrequency(frequency) {
        this.targetFrequency = Math.max(0.5, Math.min(40, frequency));
    }
    
    setMethod(method) {
        this.method = method;
    }
}
```

### **Usage Examples**

```javascript
// Example 1: Generate binaural beat for meditation (10Hz alpha)
const meditationAudio = generateBinauralBeat(originalAudio, 10, 300);

// Example 2: Generate isochronic tone for focus (20Hz beta)
const focusAudio = generateIsochronicTone(originalAudio, 20, 250);

// Example 2.5: Generate monaural beat for relaxation (8Hz alpha)
const relaxationAudio = generateMonauralBeat(originalAudio, 8, 300);

// Example 3: Progressive entrainment from alert to deep meditation
const progressiveAudio = generateRampingBinaural(originalAudio, 20, 6, 300); // 20Hz to 6Hz over 5 minutes

// Example 4: Real-time processing with monaural beats
const processor = new EntrainmentProcessor(44100);
processor.setTargetFrequency(8); // Theta for creativity
processor.setMethod('monaural');

// In audio processing loop:
const processedBuffer = processor.processBuffer(inputBuffer);
```

## Frequency Shifting Analysis

### **What Frequency Shifting Does**
When you shift the entire audio spectrum by 10Hz and create a binaural presentation, you're essentially:
- **Left channel**: Original audio (e.g., 440Hz stays 440Hz)
- **Right channel**: Frequency-shifted audio (e.g., 440Hz → 450Hz)
- Creating a **complex binaural beat** effect across the entire frequency spectrum
- Preserving the harmonic relationships within each channel separately

### **Traditional Binaural Beats vs. Frequency-Shifted Binaural**

**Traditional Binaural Beats:**
- Pure carrier tones with small frequency differences (e.g., 300Hz left, 310Hz right)
- Creates single, focused 10Hz beat frequency in the brain
- Well-researched with established effectiveness
- Simple, predictable entrainment signal

**Frequency-Shifted Binaural Approach:**
- Entire audio spectrum shifted by target frequency (e.g., 10Hz shift)
- Creates multiple simultaneous beat frequencies across all spectral components
- Every frequency component contributes to the 10Hz difference pattern
- More complex, potentially more immersive entrainment experience

### **Literature Perspective**

**Limited Direct Research:**
- Most brainwave entrainment studies focus on pure tones or traditional binaural/isochronic methods
- Few studies specifically examine whole-spectrum frequency shifting for entrainment
- The research that exists is more in audio processing than neuroscience

**Theoretical Considerations:**

**Potential Benefits:**
- **Harmonic preservation**: Each channel maintains musical coherence individually
- **Rich entrainment signal**: Every frequency component contributes to the target beat frequency
- **Immersive experience**: Full-spectrum binaural beats may create stronger entrainment
- **Natural audio quality**: No added tones, just the original content in a different presentation

**Potential Limitations:**
- **Pitch perception changes**: Right channel may sound "off-key" or unnatural due to frequency shift
- **Complex processing**: Brain must process multiple simultaneous beat frequencies
- **Potential masking**: Some beat frequencies may interfere with or mask others
- **Unknown effectiveness**: Limited research on full-spectrum frequency shifting for entrainment

### **What Research Suggests Works Better**

**Established Methods:**
1. **Pure carrier tones** (200-400Hz) with precise frequency differences
2. **Focused frequency bands** rather than broadband shifting
3. **Consistent, predictable rhythms** for reliable entrainment

**Alternative Approaches with Better Evidence:**
- **Traditional monaural beats** using pure carrier tones
- **Amplitude modulation** of existing audio at target frequency
- **Rhythmic gating** (turning audio on/off at entrainment rate)
- **Adding subliminal pure tones** beneath the audio mix

### **Frequency-Shifted Binaural Implementation**

```javascript
function generateFrequencyShiftedBinaural(originalAudio, shiftFrequency) {
    const sampleRate = originalAudio.sampleRate;
    const duration = originalAudio.duration;
    
    // Left channel: Original audio (unchanged)
    const leftChannel = originalAudio.left.slice(); // Copy original
    
    // Right channel: Frequency-shifted version
    const rightChannel = frequencyShiftAudio(originalAudio.right, shiftFrequency, sampleRate);
    
    return { left: leftChannel, right: rightChannel };
}

function frequencyShiftAudio(audioData, shiftHz, sampleRate) {
    // This is a simplified conceptual implementation
    // Real frequency shifting requires complex signal processing (Hilbert transforms, etc.)
    
    const shiftedData = new Float32Array(audioData.length);
    const phaseIncrement = 2 * Math.PI * shiftHz / sampleRate;
    let phase = 0;
    
    for (let i = 0; i < audioData.length; i++) {
        // Simplified frequency shifting using modulation
        // Note: This is not true frequency shifting, which requires more complex DSP
        const modulator = Math.cos(phase);
        shiftedData[i] = audioData[i] * modulator;
        phase += phaseIncrement;
    }
    
    return shiftedData;
}

```

### **Accurate Frequency Shifting with Python DSP**

**Required Dependencies:**
```bash
pip install numpy scipy soundfile
# Optional for advanced analysis:
pip install matplotlib librosa
```

```python
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import soundfile as sf

def accurate_frequency_shift_hilbert(audio_data, shift_hz, sample_rate):
    """
    Accurate frequency shifting using Hilbert transform method.
    This preserves the audio quality while shifting all frequencies uniformly.
    """
    # Convert to analytic signal using Hilbert transform
    analytic_signal = signal.hilbert(audio_data)
    
    # Create time vector
    duration = len(audio_data) / sample_rate
    t = np.linspace(0, duration, len(audio_data), endpoint=False)
    
    # Create complex exponential for frequency shift
    shift_exponential = np.exp(1j * 2 * np.pi * shift_hz * t)
    
    # Apply frequency shift by multiplying with complex exponential
    shifted_analytic = analytic_signal * shift_exponential
    
    # Take real part to get the shifted audio
    shifted_audio = np.real(shifted_analytic)
    
    return shifted_audio

def accurate_frequency_shift_fft(audio_data, shift_hz, sample_rate):
    """
    Alternative FFT-based frequency shifting method.
    More computationally intensive but very accurate.
    """
    # Get FFT of input signal
    fft_data = fft(audio_data)
    freqs = fftfreq(len(audio_data), 1/sample_rate)
    
    # Calculate new frequency bins after shifting
    new_freqs = freqs + shift_hz
    
    # Create new FFT array with shifted frequencies
    shifted_fft = np.zeros_like(fft_data, dtype=complex)
    
    for i, old_freq in enumerate(freqs):
        new_freq = old_freq + shift_hz
        
        # Find closest bin for new frequency
        if abs(new_freq) < sample_rate/2:  # Avoid aliasing
            new_bin = np.argmin(np.abs(freqs - new_freq))
            shifted_fft[new_bin] += fft_data[i]
    
    # Convert back to time domain
    shifted_audio = np.real(ifft(shifted_fft))
    
    return shifted_audio

def generate_frequency_shifted_binaural(audio_file_path, shift_hz, output_path=None):
    """
    Complete function to generate frequency-shifted binaural audio from a file.
    """
    # Load audio file
    audio_data, sample_rate = sf.read(audio_file_path)
    
    # Handle mono vs stereo input
    if len(audio_data.shape) == 1:
        # Mono input - duplicate to both channels
        left_channel = audio_data.copy()
        right_channel = audio_data.copy()
    else:
        # Stereo input - use existing channels
        left_channel = audio_data[:, 0]
        right_channel = audio_data[:, 1]
    
    # Left channel: Original audio (unchanged)
    output_left = left_channel
    
    # Right channel: Frequency-shifted version
    output_right = accurate_frequency_shift_hilbert(right_channel, shift_hz, sample_rate)
    
    # Combine channels
    binaural_output = np.column_stack((output_left, output_right))
    
    # Save output if path provided
    if output_path:
        sf.write(output_path, binaural_output, sample_rate)
    
    return binaural_output, sample_rate

def advanced_frequency_shift_with_filtering(audio_data, shift_hz, sample_rate, 
                                          filter_order=5, transition_width=0.1):
    """
    Advanced frequency shifting with anti-aliasing filtering.
    Includes low-pass filtering to prevent aliasing artifacts.
    """
    # Apply anti-aliasing filter before shifting
    nyquist = sample_rate / 2
    cutoff = nyquist * (1 - transition_width)
    
    # Design anti-aliasing filter
    sos = signal.butter(filter_order, cutoff, btype='low', fs=sample_rate, output='sos')
    filtered_audio = signal.sosfilt(sos, audio_data)
    
    # Apply frequency shift
    shifted_audio = accurate_frequency_shift_hilbert(filtered_audio, shift_hz, sample_rate)
    
    # Optional: Apply post-shift filtering to clean up artifacts
    post_filtered = signal.sosfilt(sos, shifted_audio)
    
    return post_filtered

def batch_process_frequency_shifts(input_file, shift_frequencies, output_dir):
    """
    Generate multiple frequency-shifted versions for testing different entrainment frequencies.
    """
    import os
    
    audio_data, sample_rate = sf.read(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    results = {}
    
    for shift_hz in shift_frequencies:
        output_path = os.path.join(output_dir, f"{base_name}_shift_{shift_hz}Hz.wav")
        binaural_audio, _ = generate_frequency_shifted_binaural(input_file, shift_hz, output_path)
        results[shift_hz] = output_path
        print(f"Generated: {output_path}")
    
    return results

# Example usage and testing
def example_usage():
    """
    Example of how to use the frequency shifting functions.
    """
    # Example 1: Basic frequency shifting for 10Hz alpha entrainment
    input_file = "meditation_audio.wav"
    output_file = "meditation_binaural_10hz.wav"
    
    binaural_audio, sr = generate_frequency_shifted_binaural(
        input_file, 
        shift_hz=10, 
        output_path=output_file
    )
    
    # Example 2: Generate multiple frequencies for testing
    shift_frequencies = [6, 8, 10, 13, 20, 40]  # Theta, alpha, beta, gamma
    results = batch_process_frequency_shifts(
        input_file, 
        shift_frequencies, 
        "output_directory"
    )
    
    # Example 3: Advanced processing with filtering
    audio_data, sample_rate = sf.read(input_file)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Use first channel for mono processing
    
    shifted_clean = advanced_frequency_shift_with_filtering(
        audio_data, 
        shift_hz=10, 
        sample_rate=sample_rate,
        filter_order=6,
        transition_width=0.05
    )
    
    return binaural_audio, results, shifted_clean

# Quality assessment function
def assess_shift_quality(original, shifted, sample_rate, shift_hz):
    """
    Assess the quality of frequency shifting by analyzing the spectrum.
    """
    from scipy import signal as sig
    
    # Compute spectrograms
    f_orig, t_orig, Sxx_orig = sig.spectrogram(original, sample_rate)
    f_shift, t_shift, Sxx_shift = sig.spectrogram(shifted, sample_rate)
    
    # Find peak frequencies in original
    orig_peaks = []
    for i in range(Sxx_orig.shape[1]):
        spectrum_slice = Sxx_orig[:, i]
        peaks, _ = sig.find_peaks(spectrum_slice, height=np.max(spectrum_slice) * 0.1)
        if len(peaks) > 0:
            orig_peaks.extend(f_orig[peaks])
    
    # Find peak frequencies in shifted
    shift_peaks = []
    for i in range(Sxx_shift.shape[1]):
        spectrum_slice = Sxx_shift[:, i]
        peaks, _ = sig.find_peaks(spectrum_slice, height=np.max(spectrum_slice) * 0.1)
        if len(peaks) > 0:
            shift_peaks.extend(f_shift[peaks])
    
    # Calculate average frequency shift
    if orig_peaks and shift_peaks:
        avg_orig = np.mean(orig_peaks)
        avg_shift = np.mean(shift_peaks)
        measured_shift = avg_shift - avg_orig
        
        print(f"Target shift: {shift_hz} Hz")
        print(f"Measured shift: {measured_shift:.2f} Hz")
        print(f"Accuracy: {(1 - abs(measured_shift - shift_hz) / shift_hz) * 100:.1f}%")
    
    return orig_peaks, shift_peaks
```

### **Recommendation**
The frequency-shifted binaural approach is theoretically interesting and could potentially create more immersive entrainment than traditional methods. However, it requires sophisticated DSP techniques for clean implementation and lacks research validation. For immediate implementation, **traditional binaural beats**, **monaural beats**, or **amplitude modulation** are more effective and easier to implement. Consider the frequency-shifted approach as an experimental feature to test against proven methods.

## Next Steps for Implementation

1. **Start with proven methods**: Implement traditional binaural beats, monaural beats, or isochronic tones
2. **Test amplitude modulation**: Apply entrainment frequency modulation to existing audio
3. **Experiment with frequency-shifted binaural**: Implement the original/shifted channel approach
4. **A/B test approaches**: Compare frequency-shifted binaural vs. traditional methods
5. **Monitor effectiveness**: Use EEG feedback to validate entrainment success across all methods
6. **Iterate based on results**: Refine parameters based on measured brainwave response

## Research Questions for Further Investigation

1. How does frequency-shifted binaural audio compare to traditional methods in controlled EEG studies?
2. Does full-spectrum binaural beating create stronger entrainment than single-frequency beats?
3. What is the optimal mix ratio for entrainment effectiveness vs. audio quality?
4. Do certain musical genres respond better to specific entrainment techniques?
5. How does individual variation affect entrainment response to different methods?
6. What DSP techniques provide the cleanest frequency shifting with minimal artifacts?
