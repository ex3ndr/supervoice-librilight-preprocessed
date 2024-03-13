# ✨ Supervoice Dataset

This is a preprocessed `librilight` dataset with ASR using Whisper Large model and then aligned with montreal forced aligner.

## Dataset Structure

Structure is similar to original dataset. Each file is represented in three formats: `.flac` with audio, `.txt` with text and `.TextGrid` with alignment. Top level folders are speakers, next one is a session and then the files split into up to `16 seconds`.

## Reproduction

To reproduce the dataset you need to execute the following steps:

```bash
datasets sync # Download source datasets (depends on your network)
./prepare_cut.sh # Cut the audio files (fast)
./prepare_tramscribe.sh # Transcribe the audio files using Whisper, this could take days and GPUs are needed
./prepare_align.sh # Align the transcribed files using montreal forced aligner, this could take days
./prepare_final.sh # Prepare the final datasets
```

## License

MIT