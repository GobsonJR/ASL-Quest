import { useEffect, useRef, useState, type RefObject } from "react";
import { HandLandmarker } from "@mediapipe/tasks-vision";
import { createVideoHandLandmarker } from "./handLandmarker";

export type OverlayStatus = "off" | "loading" | "ready" | "unavailable";

/** Draws live MediaPipe hand landmarks (all 21 points and the skeleton, for up to `maxHands` hands) on `overlayRef` over the
 * `videoRef` preview while `enabled`. It only reads the video element, never the recorded stream, so the overlay can never end
 * up in a clip. If MediaPipe cannot load (offline, blocked) the status becomes "unavailable" and nothing else is affected. */
export function useHandOverlay(
  videoRef: RefObject<HTMLVideoElement | null>,
  overlayRef: RefObject<HTMLCanvasElement | null>,
  enabled: boolean,
  maxHands = 2
): { status: OverlayStatus; hands: number } {
  const [status, setStatus] = useState<OverlayStatus>("off");
  const [hands, setHands] = useState(0);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const loadingRef = useRef<Promise<HandLandmarker> | null>(null);
  const unmountedRef = useRef(false);

  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      loadingRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      setStatus("off");
      setHands(0);
      return;
    }
    let cancelled = false;
    let raf = 0;
    let lastVideoTime = -1;
    let lastStamp = 0;
    let shown = 0;

    const clear = () => {
      const overlay = overlayRef.current;
      overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
    };

    const start = async () => {
      setStatus("loading");
      try {
        if (!landmarkerRef.current) {
          loadingRef.current ??= createVideoHandLandmarker(maxHands);
          const landmarker = await loadingRef.current;
          if (unmountedRef.current) {
            landmarker.close();
            return;
          }
          landmarkerRef.current = landmarker;
        }
      } catch {
        loadingRef.current = null;
        if (!cancelled) setStatus("unavailable");
        return;
      }
      if (cancelled) return;
      setStatus("ready");
      const landmarker = landmarkerRef.current!;

      const loop = () => {
        if (cancelled) return;
        raf = requestAnimationFrame(loop);
        const video = videoRef.current;
        const overlay = overlayRef.current;
        if (!video || !overlay || video.readyState < 2 || !video.videoWidth || video.currentTime === lastVideoTime) return;
        lastVideoTime = video.currentTime;
        const stamp = Math.max(performance.now(), lastStamp + 1); // strictly increasing timestamps
        lastStamp = stamp;
        let result;
        try {
          result = landmarker.detectForVideo(video, stamp);
        } catch {
          return; // a bad frame must never disturb recording
        }
        if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
        if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;
        const ctx = overlay.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        const w = overlay.width;
        const h = overlay.height;
        for (const points of result.landmarks) {
          ctx.lineWidth = Math.max(2, w / 320);
          ctx.strokeStyle = "#58cc02";
          for (const { start: a, end: b } of HandLandmarker.HAND_CONNECTIONS) {
            ctx.beginPath();
            ctx.moveTo(points[a].x * w, points[a].y * h);
            ctx.lineTo(points[b].x * w, points[b].y * h);
            ctx.stroke();
          }
          ctx.fillStyle = "#ff4d4d";
          for (const p of points) {
            ctx.beginPath();
            ctx.arc(p.x * w, p.y * h, Math.max(3, w / 190), 0, Math.PI * 2);
            ctx.fill();
          }
        }
        if (result.landmarks.length !== shown) {
          shown = result.landmarks.length;
          setHands(shown);
        }
      };
      raf = requestAnimationFrame(loop);
    };
    void start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      clear();
    };
  }, [enabled, maxHands, videoRef, overlayRef]);

  return { status, hands };
}
