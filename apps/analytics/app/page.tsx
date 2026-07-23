import { readFile } from "node:fs/promises";
import path from "node:path";
import { Dashboard } from "@/components/Dashboard";
import type { ComparisonData } from "@/lib/types";

// Do not prerender the matrix into a public static HTML artifact.
export const dynamic = "force-dynamic";

async function loadComparison(): Promise<ComparisonData> {
  const file = path.join(process.cwd(), "data", "comparison.json");
  const raw = await readFile(file, "utf8");
  return JSON.parse(raw) as ComparisonData;
}

export default async function HomePage() {
  const data = await loadComparison();
  return <Dashboard data={data} />;
}
