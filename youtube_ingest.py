from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    try:
        parsed = urlparse(url)

        # Case 1: youtube.com
        if "youtube.com" in parsed.netloc:
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]

        # Case 2: youtu.be
        if "youtu.be" in parsed.netloc:
            return parsed.path.strip("/")

    except:
        pass

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
