/**
 * Copy the latest bench comparison snapshot into public/data for the site build.
 * Source of truth: services/avatar/bench/ (models.json + results/*.json).
 */
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(appRoot, "../..");
const avatarRoot = path.join(repoRoot, "services", "avatar");
const outDir = path.join(appRoot, "data");
const outFile = path.join(outDir, "comparison.json");

fs.mkdirSync(outDir, { recursive: true });

const pyCandidates = [
  path.join(avatarRoot, ".venv", "Scripts", "python.exe"),
  path.join(avatarRoot, ".venv", "bin", "python"),
  "python",
];

const pyCode = `
import json, sys
sys.path.insert(0, r${JSON.stringify(path.join(avatarRoot, "bench"))})
from store import comparison_matrix
print(json.dumps(comparison_matrix(), indent=2))
`;

let written = false;
for (const py of pyCandidates) {
  if (py.includes(".venv") && !fs.existsSync(py)) continue;
  const r = spawnSync(py, ["-c", pyCode], { encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
  if (r.status === 0 && r.stdout && r.stdout.includes('"rows"')) {
    fs.writeFileSync(outFile, r.stdout.trim() + "\n", "utf8");
    written = true;
    console.log(`Synced comparison data via ${py}`);
    break;
  }
}

if (!written) {
  const fallback = path.join(avatarRoot, "bench", "comparison.json");
  if (fs.existsSync(fallback)) {
    fs.copyFileSync(fallback, outFile);
    console.log(`Copied fallback ${fallback}`);
  } else if (!fs.existsSync(outFile)) {
    console.error("Could not sync comparison data. Commit data/comparison.json or run with the avatar venv.");
    process.exit(1);
  } else {
    console.warn("Using existing data/comparison.json (sync skipped).");
  }
}
