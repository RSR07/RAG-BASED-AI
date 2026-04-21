from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def get_video_id(url):
    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    raise ValueError("Invalid YouTube URL")


def get_transcript_chunks(url):
    video_id = get_video_id(url)

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    chunks = []

    # ---- First build chunks properly
    for t in transcript:
        chunks.append({
            "file": video_id,
            "start": t.start,
            "end": t.start + t.duration,
            "text": t.text
        })

    # ---- THEN limit (fix)
    chunks = chunks[:200]

    return chunks