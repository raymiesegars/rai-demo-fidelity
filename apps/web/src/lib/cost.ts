/** Estimated USD cost for a demo session (GPU dominates). */
export const GPU_HOURLY_RATE =
  Number(process.env.NEXT_PUBLIC_GPU_HOURLY_RATE) || 0.69;

const TTS_COST_PER_CHAR = 50 / 1_000_000; // ~$50 per 1M Cartesia credits
const LLM_COST_PER_TURN = 0.00025; // ~gpt-4o-mini, rough average per turn

export function estimateSessionCostUsd(
  sessionMinutes: number,
  turnCount: number,
  ttsCharacters: number,
): {
  gpu: number;
  tts: number;
  llm: number;
  total: number;
} {
  const gpu = (sessionMinutes / 60) * GPU_HOURLY_RATE;
  const tts = ttsCharacters * TTS_COST_PER_CHAR;
  const llm = turnCount * LLM_COST_PER_TURN;
  const total = gpu + tts + llm;
  return { gpu, tts, llm, total };
}

export function formatUsd(amount: number): string {
  if (amount < 0.01) return `$${amount.toFixed(4)}`;
  return `$${amount.toFixed(2)}`;
}
