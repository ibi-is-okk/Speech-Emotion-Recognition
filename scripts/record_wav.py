import sounddevice as sd, soundfile as sf
sr, secs = 22050, 4
print("Recording for 4 seconds... speak now")
audio = sd.rec(int(secs * sr), samplerate=sr, channels=1, dtype="float32")
sd.wait()
sf.write("my_clip.wav", audio, sr)
print("Saved my_clip.wav")