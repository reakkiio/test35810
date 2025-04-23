from webscout.Provider.TTS import ElevenlabsTTS

# Initialize the TTS provider
tts = ElevenlabsTTS()

# Generate speech from text
audio_file = tts.tts("Hello, world! This is a test of the ElevenlabsTTS text-to-speech API.", voice="Brian", verbose=True)
print(f"Audio generated at: {audio_file}")

# Save the audio to a specific location
saved_path = tts.save_audio(audio_file, destination="my_speech.mp3", verbose=True)
print(f"Audio saved to: {saved_path}")


print("Streaming audio chunks:")
chunk_count = 0
for chunk in tts.stream_audio("This is a streaming test.", voice="Brian", chunk_size=1024):
    chunk_count += 1
    print(f"Received chunk {chunk_count} with {len(chunk)} bytes")

