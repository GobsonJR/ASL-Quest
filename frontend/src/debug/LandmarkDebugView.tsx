import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import modelUrl from "../../../models/hand_landmarker.task?url";
import { API_BASE } from "../config";
import { FpsMeter, RollingStat, landmarkPayload, meanJitter, type Point3 } from "./landmarkStats";

// Development-only view: it is its own page (landmarks-debug.html) and shares no state with CameraPractice or the
// production prediction flow. One HandLandmarker lives for the whole session in VIDEO mode, so MediaPipe tracks the hand
// between frames instead of re-detecting from scratch. Detection runs in a requestAnimationFrame loop and never awaits
// the network, so the camera preview is never blocked by a prediction request.

const params = new URLSearchParams(window.location.search);
const WASM_BASE = params.get("wasm") ?? "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const MODEL_URL = params.get("model") ?? modelUrl;
const PREDICT_INTERVAL_MS = 100;

type Prediction = { label: string | null; confidence: number; lowConfidence: boolean; top: string; apiMs: number; serverMs: number | null; classifyMs: number | null; totalMs: number };

const fmt = (v: number | null | undefined, digits = 1) => (v == null ? "—" : v.toFixed(digits));

export function LandmarkDebugView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [message, setMessage] = useState<string>("");
  const [classify, setClassify] = useState(false);
  const classifyRef = useRef(false);
  classifyRef.current = classify;
  const [view, setView] = useState({ detected: false, fps: 0, detMed: null as number | null, detP95: null as number | null, jitter: null as number | null, handedness: "—", detectedPct: 0 });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predError, setPredError] = useState<string | null>(null);
  const stopRef = useRef<() => void>(() => {});

  useEffect(() => () => stopRef.current(), []);

  async function start() {
    setState("loading");
    setMessage("Loading MediaPipe…");
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      const landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 960, height: 720 }, audio: false });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      let alive = true;
      let raf = 0;
      let lastVideoTime = -1;
      let lastStamp = 0;
      let prev: Point3[] | null = null;
      let frames = 0;
      let hits = 0;
      let latest: { points: Point3[]; handedness: string | null; detectedAt: number } | null = null;
      let inFlight = false;
      let lastSent = 0;
      let lastSentDetectedAt = -1;
      const fps = new FpsMeter();
      const det = new RollingStat();
      const jitterStat = new RollingStat(30);
      let lastUi = 0;
      let curFps = 0;
      let curDetected = false;
      let curHand = "—";

      const sendLatest = () => {
        // Only ever one request in flight, always with the newest landmarks: stale frames are dropped, never queued.
        if (inFlight || !latest || latest.detectedAt === lastSentDetectedAt) return;
        const snapshot = latest;
        inFlight = true;
        lastSent = performance.now();
        lastSentDetectedAt = snapshot.detectedAt;
        const body = landmarkPayload(snapshot.points, snapshot.handedness);
        const t0 = performance.now();
        fetch(`${API_BASE}/predict_landmarks`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
          .then(async (res) => {
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail ?? `HTTP ${res.status}`);
            const done = performance.now();
            setPredError(null);
            setPrediction({
              label: data.prediction,
              confidence: data.confidence,
              lowConfidence: data.low_confidence,
              top: (data.top_predictions ?? []).map((t: { label: string; confidence: number }) => `${t.label} ${(t.confidence * 100).toFixed(0)}%`).join(", "),
              apiMs: done - t0,
              serverMs: data.timings_ms?.server_total ?? null,
              classifyMs: data.timings_ms?.classify ?? null,
              totalMs: done - snapshot.detectedAt, // from the moment this frame was captured/detected to the prediction being available
            });
          })
          .catch((err) => setPredError(err instanceof Error ? err.message : String(err)))
          .finally(() => {
            inFlight = false;
          });
      };

      const loop = () => {
        if (!alive) return;
        raf = requestAnimationFrame(loop);
        if (video.readyState < 2 || video.currentTime === lastVideoTime) return; // only process genuinely new camera frames
        lastVideoTime = video.currentTime;
        const now = performance.now();
        const stamp = Math.max(now, lastStamp + 1); // MediaPipe requires strictly increasing timestamps
        lastStamp = stamp;
        const t0 = performance.now();
        const result = landmarker.detectForVideo(video, stamp);
        const detMs = performance.now() - t0;
        det.add(detMs);
        curFps = fps.tick(now);
        frames += 1;

        const overlay = overlayRef.current!;
        if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
        if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
        const ctx = overlay.getContext("2d")!;
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const hand = result.landmarks[0];
        curDetected = Boolean(hand);
        if (hand) {
          hits += 1;
          const points = hand.map((p) => ({ x: p.x, y: p.y, z: p.z }));
          const handedness = result.handedness[0]?.[0]?.categoryName ?? null;
          curHand = handedness ?? "—";
          if (prev) {
            const j = meanJitter(prev, points);
            if (j != null) jitterStat.add(j);
          }
          prev = points;
          latest = { points, handedness, detectedAt: now };
          const w = overlay.width;
          const h = overlay.height;
          ctx.lineWidth = Math.max(2, w / 320);
          ctx.strokeStyle = "#58cc02";
          for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
            ctx.beginPath();
            ctx.moveTo(points[start].x * w, points[start].y * h);
            ctx.lineTo(points[end].x * w, points[end].y * h);
            ctx.stroke();
          }
          ctx.fillStyle = "#ff4d4d";
          for (const p of points) {
            ctx.beginPath();
            ctx.arc(p.x * w, p.y * h, Math.max(3, w / 190), 0, Math.PI * 2);
            ctx.fill();
          }
          if (classifyRef.current && now - lastSent >= PREDICT_INTERVAL_MS) sendLatest();
        } else {
          prev = null;
          latest = null;
        }

        if (now - lastUi > 250) {
          lastUi = now;
          setView({ detected: curDetected, fps: curFps, detMed: det.median, detP95: det.p95, jitter: jitterStat.median, handedness: curHand, detectedPct: frames ? (100 * hits) / frames : 0 });
        }
      };
      raf = requestAnimationFrame(loop);

      stopRef.current = () => {
        alive = false;
        cancelAnimationFrame(raf);
        stream.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
        landmarker.close();
        stopRef.current = () => {};
      };
      setState("running");
      setMessage("");
    } catch (err) {
      stopRef.current();
      setState("error");
      setMessage(err instanceof Error ? err.message : String(err));
    }
  }

  function stop() {
    stopRef.current();
    setState("idle");
  }

  return (
    <main style={{ maxWidth: 980, margin: "0 auto", padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 20, fontWeight: 700 }}>Hand landmark debug (development only)</h1>
      <p style={{ fontSize: 13, opacity: 0.75 }}>
        Persistent MediaPipe HandLandmarker, VIDEO mode, running in your browser. Unmirrored preview, same orientation the backend sees.
      </p>
      <div style={{ display: "flex", gap: 8, margin: "8px 0" }}>
        {state !== "running" ? (
          <button onClick={start} disabled={state === "loading"}>{state === "loading" ? "Loading…" : "Start camera"}</button>
        ) : (
          <button onClick={stop}>Stop</button>
        )}
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input type="checkbox" checked={classify} onChange={(e) => setClassify(e.target.checked)} disabled={state !== "running"} />
          Send landmarks to /predict_landmarks (experimental classifier)
        </label>
      </div>
      {state === "error" && <p role="alert" style={{ color: "#b91c1c" }}>{message}</p>}
      <div style={{ position: "relative", background: "#000", borderRadius: 12, overflow: "hidden", aspectRatio: "4 / 3" }}>
        <video ref={videoRef} playsInline muted style={{ width: "100%", height: "100%", objectFit: "cover" }} aria-label="Webcam preview" />
        <canvas ref={overlayRef} aria-hidden="true" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
        <div role="status" style={{ position: "absolute", left: 10, top: 10, padding: "4px 10px", borderRadius: 999, color: "#fff", fontWeight: 700, background: state !== "running" ? "#555" : view.detected ? "#2e8b00" : "#b45309" }}>
          {state !== "running" ? "Camera off" : view.detected ? "Hand detected" : "No hand detected"}
        </div>
      </div>
      <section aria-label="Measurements" style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, lineHeight: 1.7, marginTop: 12 }}>
        <div>Detection FPS: {fmt(view.fps)} · detection latency median {fmt(view.detMed)} ms / p95 {fmt(view.detP95)} ms</div>
        <div>Frames with a hand: {fmt(view.detectedPct, 0)}% · MediaPipe handedness (raw): {view.handedness} · landmark jitter (median, normalised): {view.jitter == null ? "—" : view.jitter.toFixed(4)}</div>
        {classify && (
          <>
            <div>Prediction: {prediction ? `${prediction.label} ${(prediction.confidence * 100).toFixed(0)}%${prediction.lowConfidence ? " (low confidence)" : ""}` : "—"} {prediction ? `· top: ${prediction.top}` : ""}</div>
            <div>
              API round trip {fmt(prediction?.apiMs)} ms · server total {fmt(prediction?.serverMs)} ms · classifier {fmt(prediction?.classifyMs)} ms · capture→prediction {fmt(prediction?.totalMs)} ms
            </div>
            {predError && <div role="alert" style={{ color: "#b91c1c" }}>Prediction error: {predError}</div>}
          </>
        )}
      </section>
    </main>
  );
}
