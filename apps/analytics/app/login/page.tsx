"use client";

import { FormEvent, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        setError(j.error || "Login failed");
        setBusy(false);
        return;
      }
      const next = params.get("next") || "/";
      router.replace(next);
      router.refresh();
    } catch {
      setError("Network error");
      setBusy(false);
    }
  }

  return (
    <form className="login-card" onSubmit={onSubmit}>
      <h1>Model analytics</h1>
      <p>Enter the shared password to view the talking-head comparison report.</p>
      <label htmlFor="password">Password</label>
      <input
        id="password"
        name="password"
        type="password"
        autoFocus
        autoComplete="current-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={busy}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? "login-error" : undefined}
      />
      <button type="submit" disabled={busy || !password}>
        {busy ? "Checking…" : "Enter"}
      </button>
      <div className="login-error" id="login-error" role="alert" aria-live="polite">
        {error}
      </div>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="login-page">
      <Suspense fallback={<div className="login-card">Loading…</div>}>
        <LoginForm />
      </Suspense>
    </div>
  );
}
