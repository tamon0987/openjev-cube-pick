"""Voice instructions: microphone -> utterances -> transcripts -> task lists.

``listen``: VAD-segmented microphone (or file) audio sent to an OpenAI-compatible transcription server
(``scripts/stt_server.sh``). ``intent``: an utterance plus the robot's state -> a task list (gpt-5.5).
``console``: the same intents from typed lines, to test interruption without audio.
"""
