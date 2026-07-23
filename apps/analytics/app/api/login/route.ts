import { NextResponse } from "next/server";
import {
  expectedPassword,
  passwordsMatch,
  sessionCookieOptions,
  sessionToken,
} from "@/lib/auth";

export async function POST(req: Request) {
  let body: { password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid body" }, { status: 400 });
  }

  const password = (body.password || "").trim();
  const expected = expectedPassword();
  if (!expected) {
    return NextResponse.json(
      { error: "Server misconfigured (SITE_PASSWORD)" },
      { status: 500 },
    );
  }
  if (!password || !passwordsMatch(password, expected)) {
    return NextResponse.json({ error: "Incorrect password" }, { status: 401 });
  }

  let token: string;
  try {
    token = await sessionToken();
  } catch {
    return NextResponse.json(
      { error: "Server misconfigured (AUTH_SECRET)" },
      { status: 500 },
    );
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.set(sessionCookieOptions(token));
  return res;
}
