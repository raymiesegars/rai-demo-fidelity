import {
  AccessToken,
  AgentDispatchClient,
  RoomAgentDispatch,
  RoomServiceClient,
} from "livekit-server-sdk";
import { NextRequest, NextResponse } from "next/server";

const AGENT_NAME = "patient-agent";

function toApiHost(livekitUrl: string): string {
  return livekitUrl.replace(/^wss:/, "https:").replace(/^ws:/, "http:");
}

export async function GET(req: NextRequest) {
  const room =
    req.nextUrl.searchParams.get("room") ??
    process.env.NEXT_PUBLIC_LIVEKIT_ROOM ??
    "patient-demo";
  const identity =
    req.nextUrl.searchParams.get("identity") ??
    `user-${Math.random().toString(36).slice(2, 9)}`;

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;
  const livekitUrl = process.env.LIVEKIT_URL ?? process.env.NEXT_PUBLIC_LIVEKIT_URL;

  if (!apiKey || !apiSecret || !livekitUrl) {
    return NextResponse.json(
      { error: "LiveKit credentials not configured in apps/web/.env.local" },
      { status: 500 },
    );
  }

  const apiHost = toApiHost(livekitUrl);
  const roomService = new RoomServiceClient(apiHost, apiKey, apiSecret);
  const dispatchClient = new AgentDispatchClient(apiHost, apiKey, apiSecret);

  try {
    await roomService.createRoom({
      name: room,
      emptyTimeout: 600,
      departureTimeout: 120,
      agents: [new RoomAgentDispatch({ agentName: AGENT_NAME })],
    });
  } catch {
    // Room may already exist — dispatch agent again for this session.
  }

  try {
    await dispatchClient.createDispatch(room, AGENT_NAME);
  } catch (err) {
    console.warn("Agent dispatch:", err);
  }

  const token = new AccessToken(apiKey, apiSecret, {
    identity,
    name: "Demo User",
  });
  token.addGrant({
    roomJoin: true,
    room,
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
  });

  return NextResponse.json({
    token: await token.toJwt(),
    room,
    url: livekitUrl,
  });
}
