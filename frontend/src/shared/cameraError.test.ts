import { describe, expect, it } from "vitest";
import { describeCameraError } from "./cameraError";

function domException(name: string): DOMException {
  return new DOMException("simulated", name);
}

describe("describeCameraError", () => {
  it("reports permission-denied specifically", () => {
    expect(describeCameraError(domException("NotAllowedError"))).toMatch(/denied/i);
    expect(describeCameraError(domException("PermissionDeniedError"))).toMatch(/denied/i);
  });

  it("reports no-camera-found specifically", () => {
    expect(describeCameraError(domException("NotFoundError"))).toMatch(/no camera/i);
    expect(describeCameraError(domException("DevicesNotFoundError"))).toMatch(/no camera/i);
  });

  it("reports camera-in-use specifically", () => {
    expect(describeCameraError(domException("NotReadableError"))).toMatch(/in use/i);
    expect(describeCameraError(domException("TrackStartError"))).toMatch(/in use/i);
  });

  it("reports unsupported constraints specifically", () => {
    expect(describeCameraError(domException("OverconstrainedError"))).toMatch(/settings/i);
  });

  it("falls back to a generic message for unknown/non-DOMException failures", () => {
    expect(describeCameraError(new Error("boom"))).toMatch(/isn't available/i);
    expect(describeCameraError(undefined)).toMatch(/isn't available/i);
    expect(describeCameraError(domException("SomeOtherError"))).toMatch(/isn't available/i);
  });

  it("never leaks the raw exception message", () => {
    const message = describeCameraError(new Error("ENOENT: /var/secret/path stack trace at foo.js:42"));
    expect(message).not.toContain("ENOENT");
    expect(message).not.toContain("foo.js");
  });
});
