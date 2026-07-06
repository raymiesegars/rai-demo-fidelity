"use client";

import {
  RoomAudioRenderer,
  RoomContext,
  VideoTrack,
  useTracks,
} from "@livekit/components-react";
import { ConnectionState, Room, RoomEvent, Track } from "livekit-client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  estimateSessionCostUsd,
  formatUsd,
  GPU_HOURLY_RATE,
} from "@/lib/cost";

type ChatMessage = {
  id: string;
  role: "user" | "patient" | "system";
  text: string;
};

type SessionStats = {
  turnCount: number;
  ttsCharacters: number;
};

function PatientVideo() {
  const tracks = useTracks(
    [{ source: Track.Source.Camera, withPlaceholder: false }],
    { onlySubscribed: true },
  );

  const avatarTrack = useMemo(
    () =>
      tracks.find(
        (t) =>
          t.publication &&
          (t.participant.identity === "avatar-patient" ||
            t.participant.name?.toLowerCase().includes("alan")),
      ),
    [tracks],
  );

  if (avatarTrack?.publication) {
    return (
      <VideoTrack
        trackRef={avatarTrack}
        className="h-full w-full object-cover"
      />
    );
  }

  return (
    <video
      className="h-full w-full object-cover"
      src="/alan-loop.mp4"
      autoPlay
      loop
      muted
      playsInline
    />
  );
}

function ConnectionBadge({ state }: { state: ConnectionState }) {
  const label =
    state === ConnectionState.Connected
      ? "Connected"
      : state === ConnectionState.Connecting
        ? "Connecting…"
        : state === ConnectionState.Reconnecting
          ? "Reconnecting…"
          : "Offline";

  const color =
    state === ConnectionState.Connected
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
      : "bg-amber-500/20 text-amber-200 border-amber-500/30";

  return (
    <span
      className={`rounded-full border px-3 py-1 text-xs font-medium ${color}`}
    >
      {label}
    </span>
  );
}

function CostMeter({
  sessionStart,
  stats,
}: {
  sessionStart: number | null;
  stats: SessionStats;
}) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!sessionStart) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [sessionStart]);

  const minutes = sessionStart ? (now - sessionStart) / 60_000 : 0;
  const cost = estimateSessionCostUsd(minutes, stats.turnCount, stats.ttsCharacters);

  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-sm backdrop-blur-sm">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Session cost estimate
      </p>
      <p className="text-2xl font-semibold text-white tabular-nums">
        {formatUsd(cost.total)}
      </p>
      <div className="mt-3 space-y-1 text-xs text-slate-400">
        <div className="flex justify-between">
          <span>GPU @ ${GPU_HOURLY_RATE}/hr</span>
          <span className="tabular-nums text-slate-300">{formatUsd(cost.gpu)}</span>
        </div>
        <div className="flex justify-between">
          <span>TTS ({stats.ttsCharacters.toLocaleString()} chars)</span>
          <span className="tabular-nums text-slate-300">{formatUsd(cost.tts)}</span>
        </div>
        <div className="flex justify-between">
          <span>LLM ({stats.turnCount} turns)</span>
          <span className="tabular-nums text-slate-300">{formatUsd(cost.llm)}</span>
        </div>
        <div className="flex justify-between border-t border-white/10 pt-1">
          <span>Elapsed</span>
          <span className="tabular-nums text-slate-300">
            {minutes.toFixed(1)} min
          </span>
        </div>
      </div>
    </div>
  );
}

function SessionPanel({
  room,
  onDisconnect,
}: {
  room: Room;
  onDisconnect: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "system",
      text: "Session active. Type below to talk with Alan.",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [patientSpeaking, setPatientSpeaking] = useState(false);
  const [stats, setStats] = useState<SessionStats>({
    turnCount: 0,
    ttsCharacters: 0,
  });
  const [sessionStart] = useState(() => Date.now());
  const scrollRef = useRef<HTMLDivElement>(null);
  const connectionState = room.state;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const onData = (
      payload: Uint8Array,
      _participant?: { identity: string },
      _kind?: unknown,
      topic?: string,
    ) => {
      if (topic !== "agent_reply" && topic !== "agent_error") return;
      try {
        const data = JSON.parse(new TextDecoder().decode(payload)) as {
          text?: string;
          charCount?: number;
          type?: string;
        };
        if (data.text) {
          const replyText = data.text;
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: topic === "agent_error" ? "system" : ("patient" as const),
              text: topic === "agent_error" ? `⚠ ${replyText}` : replyText,
            },
          ]);
          if (topic === "agent_reply" && data.charCount) {
            setStats((s) => ({
              turnCount: s.turnCount + 1,
              ttsCharacters: s.ttsCharacters + data.charCount!,
            }));
          }
        }
      } catch {
        /* ignore malformed */
      }
    };

    room.on(RoomEvent.DataReceived, onData);
    room.on(RoomEvent.ActiveSpeakersChanged, (speakers) => {
      const agentSpeaking = speakers.some(
        (p) =>
          p.identity.includes("agent") ||
          p.identity.includes("patient-agent"),
      );
      setPatientSpeaking(agentSpeaking);
    });

    return () => {
      room.off(RoomEvent.DataReceived, onData);
    };
  }, [room]);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    setSending(true);
    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text },
    ]);

    const payload = new TextEncoder().encode(JSON.stringify({ text }));
    await room.localParticipant.publishData(payload, {
      reliable: true,
      topic: "user_text",
    });
    setSending(false);
  }, [input, room, sending]);

  return (
    <div className="grid min-h-0 flex-1 gap-6 lg:grid-cols-[1fr_340px]">
      <div className="flex min-h-0 flex-col gap-4">
        <div className="relative aspect-[4/5] max-h-[70vh] overflow-hidden rounded-2xl border border-white/10 bg-slate-900 shadow-2xl shadow-black/40 lg:aspect-auto lg:min-h-[480px]">
          <PatientVideo />
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent" />
          <div className="absolute left-4 top-4 flex items-center gap-2">
            <ConnectionBadge state={connectionState} />
            {patientSpeaking && (
              <span className="rounded-full border border-sky-500/30 bg-sky-500/20 px-3 py-1 text-xs font-medium text-sky-200">
                Alan is speaking
              </span>
            )}
          </div>
          <div className="absolute bottom-4 left-4 right-4">
            <h2 className="text-xl font-semibold text-white">Alan</h2>
            <p className="text-sm text-slate-300">Patient · Medical intake demo</p>
          </div>
        </div>
      </div>

      <div className="flex min-h-0 flex-col gap-4">
        <CostMeter sessionStart={sessionStart} stats={stats} />

        <div className="flex min-h-0 flex-1 flex-col rounded-2xl border border-white/10 bg-slate-900/80 backdrop-blur-sm">
          <div
            ref={scrollRef}
            className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4"
          >
            {messages.map((m) => (
              <div
                key={m.id}
                className={
                  m.role === "user"
                    ? "ml-8 rounded-2xl rounded-br-md bg-sky-600/90 px-4 py-2.5 text-sm text-white"
                    : m.role === "patient"
                      ? "mr-8 rounded-2xl rounded-bl-md border border-white/10 bg-white/5 px-4 py-2.5 text-sm text-slate-100"
                      : "text-center text-xs text-slate-500"
                }
              >
                {m.text}
              </div>
            ))}
          </div>

          <form
            className="border-t border-white/10 p-3"
            onSubmit={(e) => {
              e.preventDefault();
              void sendMessage();
            }}
          >
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message to Alan…"
                disabled={connectionState !== ConnectionState.Connected}
                className="flex-1 rounded-xl border border-white/10 bg-slate-950/60 px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:border-sky-500/50 focus:outline-none focus:ring-1 focus:ring-sky-500/30 disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={
                  !input.trim() ||
                  sending ||
                  connectionState !== ConnectionState.Connected
                }
                className="rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Send
              </button>
            </div>
          </form>
        </div>

        <button
          type="button"
          onClick={onDisconnect}
          className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-400 transition hover:border-red-500/30 hover:text-red-300"
        >
          End session
        </button>
      </div>

      <RoomAudioRenderer />
    </div>
  );
}

export default function PatientSession() {
  const [room] = useState(() => new Room({ adaptiveStream: false, dynacast: false }));
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  const connect = useCallback(async () => {
    setConnecting(true);
    setError(null);
    try {
      const res = await fetch("/api/token");
      if (!res.ok) {
        const body = (await res.json()) as { error?: string };
        throw new Error(body.error ?? "Failed to get token");
      }
      const { token, url, room: roomName } = await res.json();
      await room.connect(url, token);
      setConnected(true);
      console.info("Joined room:", roomName);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed");
    } finally {
      setConnecting(false);
    }
  }, [room]);

  const disconnect = useCallback(async () => {
    await room.disconnect();
    setConnected(false);
  }, [room]);

  useEffect(() => {
    return () => {
      void room.disconnect();
    };
  }, [room]);

  if (!connected) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-6 py-16">
        <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-slate-900/60 p-10 text-center shadow-2xl backdrop-blur-sm">
          <div className="mx-auto mb-6 h-24 w-24 overflow-hidden rounded-full border-2 border-white/20">
            <video
              className="h-full w-full object-cover"
              src="/alan-loop.mp4"
              autoPlay
              loop
              muted
              playsInline
            />
          </div>
          <h1 className="text-2xl font-semibold text-white">Talk with Alan</h1>
          <p className="mt-2 text-sm leading-relaxed text-slate-400">
            High-fidelity patient avatar demo. Connect to start a LiveKit session,
            then type to converse. Alan responds with voice and lip-synced video.
          </p>
          {error && (
            <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {error}
            </p>
          )}
          <button
            type="button"
            onClick={() => void connect()}
            disabled={connecting}
            className="mt-8 w-full rounded-xl bg-sky-600 py-3 text-sm font-semibold text-white transition hover:bg-sky-500 disabled:opacity-50"
          >
            {connecting ? "Connecting…" : "Start session"}
          </button>
          <p className="mt-4 text-xs text-slate-500">
            Est. ~$0.12 per 10 min @ ${GPU_HOURLY_RATE}/hr GPU
          </p>
        </div>
      </div>
    );
  }

  return (
    <RoomContext.Provider value={room}>
      <div className="flex min-h-0 flex-1 flex-col px-4 py-6 lg:px-8">
        <SessionPanel room={room} onDisconnect={() => void disconnect()} />
      </div>
    </RoomContext.Provider>
  );
}
