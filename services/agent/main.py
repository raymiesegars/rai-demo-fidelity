"""LiveKit agent — text chat in, Cartesia TTS audio out."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    llm,
)
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import cartesia, openai

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("patient-agent")

PATIENT_INSTRUCTIONS = """You are Alan, a patient in a medical intake demo.
You are calm, cooperative, and speak in short natural sentences (1-3 sentences per reply).
You have mild lower-back pain for about two weeks, worse when bending.
Answer the clinician's questions directly. Do not break character or mention being an AI.
Keep responses under 80 words unless asked for detail."""


class PatientAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=PATIENT_INSTRUCTIONS)


def build_llm() -> llm.LLM:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider == "groq":
        from livekit.plugins import groq

        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info("Using Groq LLM: %s", model)
        return groq.LLM(model=model)

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    logger.info("Using OpenAI LLM: %s", model)
    return openai.LLM(model=model)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    voice_id = os.environ.get(
        "CARTESIA_VOICE_ID", "df89f42f-f285-4613-adbf-14eedcec4c9e"
    )

    session = AgentSession(
        llm=build_llm(),
        tts=cartesia.TTS(voice=voice_id),
    )

    async def publish_to_client(payload: dict) -> None:
        await ctx.room.local_participant.publish_data(
            json.dumps(payload).encode(),
            reliable=True,
            topic="agent_reply" if payload.get("type") != "error" else "agent_error",
        )

    @session.on("conversation_item_added")
    def on_conversation_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        role = getattr(item, "role", None)
        if role != "assistant":
            return
        text = getattr(item, "text_content", None) or ""
        if not text:
            return

        asyncio.create_task(
            publish_to_client({"text": text, "charCount": len(text)})
        )

    @session.on("error")
    def on_session_error(ev: object) -> None:
        err = getattr(ev, "error", ev)
        msg = str(err)
        if "insufficient_quota" in msg or "429" in msg:
            msg = (
                "OpenAI quota exceeded. Add billing at platform.openai.com/settings/"
                "organization/billing — or set LLM_PROVIDER=groq with a free Groq key."
            )
        logger.error("Agent session error: %s", msg)
        asyncio.create_task(publish_to_client({"type": "error", "text": msg}))

    agent = PatientAgent()
    await session.start(agent=agent, room=ctx.room)

    @ctx.room.on("data_received")
    def on_data(data: rtc.DataPacket) -> None:
        if data.topic != "user_text":
            return
        try:
            message = json.loads(data.data.decode("utf-8"))
            user_text = message.get("text", "").strip()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not user_text:
            return

        logger.info("User text: %s", user_text[:120])
        session.generate_reply(user_input=user_text, input_modality="text")

    logger.info("Patient agent ready in room %s", ctx.room.name)


if __name__ == "__main__":
    cli.run_app(server)
