# Avatar Model Analytics (Vercel)

Password-protected, read-only comparison site. Data is synced from
`services/avatar/bench/` at build time.

## Local

```powershell
cd apps\analytics
copy .env.example .env.local
# set SITE_PASSWORD and AUTH_SECRET in .env.local
npm install
npm run dev
```

Open http://localhost:3000 — you will be prompted for `SITE_PASSWORD`.

## Deploy on Vercel

1. Import this Git repository in Vercel.
2. Set **Root Directory** to `apps/analytics`.
3. Add environment variables:
   - `SITE_PASSWORD` — shared password visitors enter
   - `AUTH_SECRET` — long random string (`openssl rand -hex 32`)
4. Deploy. After bench scores change in the repo, redeploy (or push) so
   `npm run build` re-syncs `data/comparison.json`.

Optional: enable Vercel Deployment Protection as a second layer. This app
already requires `SITE_PASSWORD` before any page content is shown. Data lives
in `data/` (server-only), not in a public static URL.
