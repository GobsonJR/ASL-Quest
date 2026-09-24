/** Turns a getUserMedia/MediaRecorder failure into a specific, learner-facing
 * message instead of one generic "camera unavailable" string for every
 * failure type. Shared by CameraPractice (A-Z/Word Spelling) and
 * NativeVideoCapture (Native Signs) so both cameras report errors the same
 * way. Never surfaces raw exception text/stack traces to the learner. */
export function describeCameraError(err: unknown): string {
  const name = err instanceof DOMException ? err.name : null;
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera access was denied. Allow camera permissions in your browser and try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found on this device.";
    case "NotReadableError":
    case "TrackStartError":
      return "Your camera is already in use by another app. Close it and try again.";
    case "OverconstrainedError":
      return "Your camera doesn't support the required video settings.";
    default:
      return "Your camera isn't available right now. Check permissions and try again.";
  }
}
