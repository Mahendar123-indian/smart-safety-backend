import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="test_voice.wav", duration=5):
    fs = 44100  # Sample rate
    print(f"🎤 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file
    print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    # You might need to run: pip install sounddevice scipy
    record_audio()