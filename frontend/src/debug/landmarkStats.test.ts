import { describe, expect, it } from "vitest";
import { FpsMeter, RollingStat, landmarkPayload, meanJitter } from "./landmarkStats";

describe("RollingStat", () => {
  it("reports median and p95 and keeps only the latest window", () => {
    const stat = new RollingStat(5);
    expect(stat.median).toBeNull();
    [10, 20, 30, 40, 50, 60].forEach((v) => stat.add(v));
    expect(stat.count).toBe(5);
    expect(stat.median).toBe(40);
    expect(stat.p95).toBe(60);
  });
});

describe("FpsMeter", () => {
  it("computes frames per second from timestamps", () => {
    const meter = new FpsMeter(1000);
    let fps = 0;
    for (let t = 0; t <= 500; t += 50) fps = meter.tick(t);
    expect(fps).toBeCloseTo(20, 5);
  });
});

describe("meanJitter", () => {
  const hand = Array.from({ length: 21 }, (_, i) => ({ x: i / 21, y: 0.5, z: 0 }));
  it("is zero for an identical hand and positive when points move", () => {
    expect(meanJitter(hand, hand)).toBe(0);
    expect(meanJitter(hand, hand.map((p) => ({ ...p, x: p.x + 0.01 })))).toBeCloseTo(0.01, 6);
  });
  it("returns null for mismatched lengths", () => {
    expect(meanJitter(hand, hand.slice(1))).toBeNull();
  });
});

describe("landmarkPayload", () => {
  const hand = Array.from({ length: 21 }, (_, i) => ({ x: 0.1, y: 0.2, z: i / 100 }));
  it("sends 21 [x,y,z] rows and only valid MediaPipe handedness labels", () => {
    const body = landmarkPayload(hand, "Left");
    expect(body.landmarks).toHaveLength(21);
    expect(body.landmarks[3]).toEqual([0.1, 0.2, 0.03]);
    expect(body.handedness).toBe("Left");
    expect(body.image_mirrored).toBe(false);
    expect(landmarkPayload(hand, "weird").handedness).toBeNull();
  });
});
