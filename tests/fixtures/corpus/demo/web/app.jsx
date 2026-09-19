import { useState } from "react";

// The intro animation plays once, then the dashboard renders.
export default function App() {
  const [phase, setPhase] = useState("intro");
  const steps = ["fade-in", "slide", "reveal"];
  return (
    <main>
      {phase === "intro" ? <Intro steps={steps} onDone={() => setPhase("ready")} /> : <Dashboard />}
    </main>
  );
}

function Intro({ steps, onDone }) {
  return <div className="intro" onAnimationEnd={onDone}>{steps.join(" > ")}</div>;
}

function Dashboard() {
  return <section>Reports</section>;
}
