from transcription.audio_transcript_gen import transcript_whisper_timestamped, read_video, extract_audio
from HighlightDetection.TextTiling import text_tiling 
from HighlightDetection.Important_Sentence import extract_important_sentences
from clip_extraction.video_crop_utils import clip_extractor

if __name__ == "__main__":


	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    #boilerplate pytorch code enforcing reproducibility
    torch.manual_seed(10)
    if device.type == "cuda":
        torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

	fname = "...."

	vid = read_video(fname)
	aud = extract_audio(vid)


	[aligned_text_transcribedTS, aligned_segmentsTS] = transcript_whisper_timestamped("test_audio.mp3", device=device, model_type ="large-v1" , translate=False)

	segments = text_tiling(aligned_text_transcribedTS)

	highlights = []

	for i in segments:
		highlights.append(extract_important_sentences(i))
	

	clip = clip_extractor(fname, segments, aligned_text_transcribedTS, highlights)
	clips = clip.get_clip(clip.get_timestamps())