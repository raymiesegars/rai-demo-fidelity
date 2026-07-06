import PatientSession from "@/components/PatientSession";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <header className="border-b border-white/5 px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-sky-400/80">
              RAI Demo
            </p>
            <h1 className="text-lg font-semibold text-white">Patient Fidelity</h1>
          </div>
          <p className="hidden text-xs text-slate-500 sm:block">
            LiveKit · OpenAI · Cartesia · RunPod
          </p>
        </div>
      </header>
      <PatientSession />
    </main>
  );
}
