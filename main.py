# Importing Necessary Libraries
import os
import re
import subprocess
from pathlib import Path
import asyncio
import logging
import assemblyai as aai
import cv2
import yt_dlp as youtube_dl
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from typing import Dict

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mount the main static directory (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define paths for transcripts and detected images
TRANSCRIPTS_DIR = Path("C:/Users/hp/OneDrive/Desktop/My Transcripts")
DETECTED_IMAGES_DIR = Path("C:/Users/hp/OneDrive/Desktop/My Transcripts/Detected images")

# Create directories if they don't exist
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
DETECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Mount transcripts and detected_images as separate static routes
app.mount("/transcripts", StaticFiles(directory=str(TRANSCRIPTS_DIR)), name="transcripts")
app.mount("/detected_images", StaticFiles(directory=str(DETECTED_IMAGES_DIR)), name="detected_images")

# Setting the AssemblyAI API Key from environment variable for security
aai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
if not aai_api_key:
    logger.warning("AssemblyAI API key not found. Using default key for development.")
    aai_api_key = "0003f128d8cd46fd9aee533c981fdf26"  # Fallback for development/testing
aai.settings.api_key = aai_api_key
transcriber = aai.Transcriber()

# Initializing YOLO Model Once
try:
    yolo_model = YOLO('static/yolov8n.pt')  # Ensure 'yolov8n.pt' is in the 'static' directory
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

# Manage WebSocket Connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, request_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[request_id] = websocket
        logger.info(f"WebSocket connection established for request ID: {request_id}")

    def disconnect(self, request_id: str):
        if request_id in self.active_connections:
            del self.active_connections[request_id]
            logger.info(f"WebSocket connection closed for request ID: {request_id}")

    async def send_message(self, request_id: str, message: str):
        if request_id in self.active_connections:
            websocket = self.active_connections[request_id]
            await websocket.send_json({"status": message})
            logger.info(f"Sent status to {request_id}: {message}")

    async def send_completion(self, request_id: str, transcription: str, diagram_url: str, summary: str, image_paths: list):
        if request_id in self.active_connections:
            websocket = self.active_connections[request_id]
            await websocket.send_json({
                "completed": True,
                "transcription": transcription,
                "diagram_url": diagram_url,
                "summary": summary,  # Send summary back to client
                "image_paths": image_paths  # Send all image paths back to client
            })
            logger.info(f"Sent completion to {request_id}")
            await websocket.close()
            self.disconnect(request_id)

manager = ConnectionManager()

def download_video(youtube_link: str, output_path: Path) -> None:
    """Download the YouTube video using yt-dlp in .webm format."""
    ydl_opts = {
        'format': 'bestvideo[ext=webm]+bestaudio[ext=webm]/best[ext=webm]',
        'outtmpl': str(output_path),
        'merge_output_format': 'webm',
        'quiet': True,  # Suppress yt-dlp output
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            logger.info(f"Downloading video from {youtube_link}...")
            ydl.download([youtube_link])
            logger.info(f"Video downloaded successfully to {output_path}.")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise HTTPException(status_code=400, detail="Failed to download video. Please check the YouTube link.")

def download_audio(video_url: str, audio_path: Path) -> None:
    """Download the audio from the YouTube video and save as MP3."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(audio_path.with_suffix('.%(ext)s')),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            logger.info(f"Downloading audio from {video_url}...")
            ydl.download([video_url])
            logger.info(f"Audio downloaded successfully to {audio_path.with_suffix('.mp3')}.")
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise HTTPException(status_code=400, detail="Failed to download audio.")

async def convert_to_wav(mp3_path: Path, wav_path: Path) -> None:
    """Convert MP3 audio file to WAV format using ffmpeg asynchronously."""
    try:
        logger.info(f"Converting {mp3_path} to WAV format...")
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-i', str(mp3_path), str(wav_path),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            logger.error(f"FFmpeg error: {error_msg}")
            raise HTTPException(status_code=500, detail="Failed to convert audio to WAV format.")
        logger.info(f"Audio converted to WAV format at {wav_path}.")
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        raise HTTPException(status_code=500, detail="Failed to convert audio to WAV format.")

def transcribe_audio(wav_path: Path) -> str:
    """Transcribe the WAV audio file using AssemblyAI."""
    try:
        logger.info(f"Transcribing audio file {wav_path}...")
        transcript = transcriber.transcribe(str(wav_path))
        logger.info("Transcription completed.")
        return transcript.text
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio.")

def summarize_text(transcript: str, summary_length: str) -> str:
    """Summarize the transcription text based on the selected summary length."""
    # Placeholder for actual summarization logic.
    # You can integrate a summarization library or API here.
    try:
        logger.info("Generating summary...")
        words = transcript.split()
        if summary_length == "short":
            return ' '.join(words[:100]) + '...'
        elif summary_length == "medium":
            return ' '.join(words[:300]) + '...'
        elif summary_length == "long":
            return ' '.join(words[:500]) + '...'
        else:
            return transcript  # Default to full transcript
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary.")

def detect_and_save_frames(video_path: Path, output_folder: Path, unique_id: str) -> list:
    """
    Detect objects in every 200th frame of the video and save those frames.

    Args:
        video_path (Path): Path to the video file.
        output_folder (Path): Directory to save detected images.
        unique_id (str): Unique identifier for the request to prevent filename conflicts.

    Returns:
        list: List of saved image file paths.
    """
    saved_images = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {video_path}.")
        raise HTTPException(status_code=400, detail="Failed to open video file for detection.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Reached end of video.")
            break

        frame_count += 1

        # Process every 200th frame
        if frame_count % 200 == 0:
            if yolo_model is None:
                logger.error("YOLO model not loaded.")
                break

            results = yolo_model(frame)
            detections = results[0].boxes.data.tolist()  # List of detections

            # Draw bounding boxes if detections are present
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                # Define the classes you are interested in (optional)
                if int(cls) in [1, 2, 3, 14, 15, 16, 17, 18]:  # Example classes
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Save the frame with detections (regardless of whether any detections were made)
            image_filename = f"detected_frame_{unique_id}_{frame_count}.jpg"
            image_path = output_folder / image_filename
            success = cv2.imwrite(str(image_path), frame)
            if success:
                saved_images.append(str(image_path))
                logger.info(f"Saved frame {frame_count} as {image_filename}.")
            else:
                logger.error(f"Failed to save frame {frame_count}.")

    cap.release()
    logger.info(f"Detection completed. {len(saved_images)} images saved in '{output_folder}'.")
    return saved_images

def get_video_title(youtube_link: str) -> str:
    """Extract the video title from the YouTube link using yt-dlp."""
    ydl_opts = {
        'quiet': True,  # Suppress yt-dlp output
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(youtube_link, download=False)
            video_title = info_dict.get('title', None)
            logger.info(f"Extracted video title: {video_title}")
            return re.sub(r'[\\/*?:"<>|]', "", video_title)  # Clean up invalid characters for folder name
        except Exception as e:
            logger.error(f"Error extracting video title: {e}")
            raise HTTPException(status_code=400, detail="Failed to retrieve video title.")

@app.get("/", response_class=HTMLResponse)
async def get_form():
    """Serve the HTML form for users to input the YouTube URL and desired transcript name."""
    try:
        content = open('index.html').read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        logger.error("index.html not found.")
        raise HTTPException(status_code=404, detail="Page not found.")

@app.websocket("/ws/{request_id}")
async def websocket_endpoint(websocket: WebSocket, request_id: str):
    """Handle WebSocket connections."""
    await manager.connect(request_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received data from {request_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(request_id)

@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe_video(
    video_url: str = Form(...),
    transcript_name: str = Form(...),
    summary_length: str = Form(...),  
    language: str = Form(...),       
    request_id: str = Form(...)
):
    """Handle the video URL input, process transcription, and detect images."""
    unique_id = request_id  # Use the request ID from the form

    # Step 1: Retrieve Video Title
    video_title = get_video_title(video_url)

    # Step 2: Define paths with the video title to prevent conflicts
    video_path = Path(f"video_{unique_id}.webm")
    audio_mp3_path = Path(f"audio_{unique_id}.mp3")
    audio_wav_path = Path(f"audio_{unique_id}.wav")
    output_images_folder = DETECTED_IMAGES_DIR / video_title
    output_images_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Step 3: Download Video
        await manager.send_message(unique_id, "Downloading video...")
        download_video(video_url, video_path)

        # Step 4: Download Audio from the Video
        await manager.send_message(unique_id, "Downloading audio...")
        download_audio(video_url, audio_mp3_path)

        # Step 5: Convert MP3 to WAV
        await manager.send_message(unique_id, "Converting audio to WAV format...")
        await convert_to_wav(audio_mp3_path, audio_wav_path)

        # Step 6: Transcribe the Audio
        await manager.send_message(unique_id, "Transcribing audio...")
        transcript_text = transcribe_audio(audio_wav_path)

        # Step 7: Save the Transcript to 'My Transcripts' Directory
        transcript_file_path = TRANSCRIPTS_DIR / f"{transcript_name}.txt"
        with open(transcript_file_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        logger.info(f"Transcript saved to {transcript_file_path}.")
        await manager.send_message(unique_id, "Transcript saved.")

        # Step 8: Summarize the Transcription
        await manager.send_message(unique_id, "Generating summary...")
        summary_text = summarize_text(transcript_text, summary_length)

        # Step 9: Perform Object Detection and Save Every 200th Frame
        await manager.send_message(unique_id, "Performing object detection on video frames...")
        saved_images = detect_and_save_frames(video_path, output_images_folder, unique_id)

        # Step 10: Prepare Links to Saved Resources
        # Convert filesystem paths to URLs
        images_links = [f"/detected_images/{video_title}/{Path(img).name}" for img in saved_images]

        # Create Diagram URL (Example: Display the first detected image)
        diagram_url = images_links[0] if images_links else ""

        # Step 11: Send Completion Message via WebSocket
        await manager.send_completion(unique_id, transcript_text, diagram_url, summary_text, images_links)

        return HTMLResponse(content="")  # Empty response as the frontend is handling updates via WebSocket

    except HTTPException as he:
        await manager.send_message(unique_id, f"Error: {he.detail}")
        raise he  # Re-raise HTTP exceptions to be handled by FastAPI

    except Exception as e:
        error_message = "An unexpected error occurred during processing."
        logger.error(f"An unexpected error occurred: {e}")
        await manager.send_message(unique_id, f"Error: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)

    finally:
        # Clean up temporary files
        for temp_file in [video_path, audio_mp3_path, audio_wav_path]:
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"Deleted temporary file {temp_file}.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
