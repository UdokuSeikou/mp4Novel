import os
from pathlib import Path
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import whisper

load_dotenv('../.env')
# Set the paths
video_path = os.getenv('VIDEO_TEST_PATH')
output_audio_path = os.path.join(Path.cwd(), "temp_audio.mp3")

# Extract audio from the video
video = VideoFileClip(video_path)
video.audio.write_audiofile(output_audio_path)

# Load the Whisper ASR model
model = whisper.load_model("small")

# Transcribe the extracted audio
result = model.transcribe(output_audio_path)
print(result["text"])

# Remove the temporary audio file
os.remove(output_audio_path)

