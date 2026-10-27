"""Johnny: a conversational voice agent for the rig (experimental).

Wake word ("Hi, Johnny") -> the utterance goes to an LLM that keeps the chat history and calls skills
(set up the rig, tele-operate, pick and place with the grasp VLM + Jev, stop). Skills are plain Python modules
under ``dlb.johnny.skills``; see ``dlb/johnny/skills/base.py`` for the interface.
"""
