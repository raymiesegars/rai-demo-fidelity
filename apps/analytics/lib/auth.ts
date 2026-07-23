/** Edge-safe session helpers (Web Crypto). Used by middleware + API routes. */

export const COOKIE_NAME = "avatar_analytics_session";
const MAX_AGE_S = 60 * 60 * 24 * 14;

function secret(): string {
  return process.env.AUTH_SECRET || process.env.SITE_PASSWORD || "";
}

export function expectedPassword(): string {
  return process.env.SITE_PASSWORD || "";
}

async function sha256Base64Url(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const bytes = new Uint8Array(digest);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function hmacBase64Url(key: string, msg: string): Promise<string> {
  const enc = new TextEncoder();
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(key),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", cryptoKey, enc.encode(msg));
  const bytes = new Uint8Array(sig);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let out = 0;
  for (let i = 0; i < a.length; i++) out |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return out === 0;
}

/** Stable token derived from SITE_PASSWORD + AUTH_SECRET. */
export async function sessionToken(): Promise<string> {
  const pwd = expectedPassword();
  const sec = secret();
  if (!pwd || !sec) throw new Error("SITE_PASSWORD and AUTH_SECRET must be set");
  const bind = await sha256Base64Url(`${pwd}:${sec}`);
  return hmacBase64Url(sec, `avatar-analytics-v1:${bind}`);
}

export async function verifySession(token: string | undefined): Promise<boolean> {
  if (!token) return false;
  try {
    const expect = await sessionToken();
    return safeEqual(token, expect);
  } catch {
    return false;
  }
}

export function passwordsMatch(input: string, expected: string): boolean {
  return safeEqual(input, expected);
}

export function sessionCookieOptions(token: string) {
  return {
    name: COOKIE_NAME,
    value: token,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax" as const,
    path: "/",
    maxAge: MAX_AGE_S,
  };
}
