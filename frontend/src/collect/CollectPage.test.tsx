import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { CollectPage, collectTiming } from "./CollectPage";
import * as api from "./api";
import { predictImage } from "../api";

// Network layer mocked; the page's real logic (letter selection, review, confirmation, hold capture, undo) runs.
vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    collectStatus: vi.fn(),
    listSessions: vi.fn(),
    createSession: vi.fn(),
    analyzeFrame: vi.fn(),
    saveSample: vi.fn(),
    deleteSample: vi.fn(),
  };
});
// The collector must never use the recognizer: if the page imported it, this spy would be callable.
vi.mock("../api", () => ({ predictImage: vi.fn() }));

const LETTERS = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
const STATUS: api.CollectStatus = {
  enabled: true,
  letters: LETTERS,
  options: { lighting: ["lamp", "daylight"], background: ["shelves", "plain_wall"], hand: ["right", "left"], distance: ["medium", "near"] },
  target_per_letter: 150,
  totals: Object.fromEntries(LETTERS.map((l) => [l, 0])),
  thresholds: { min_sharpness: 110, min_hand_px: 100, brightness_range: [45, 225] },
  hand_detector_available: true,
};
const SESSION: api.SessionInfo = {
  session_id: "s20261019_120000_room",
  room: "room", lighting: "lamp", background: "shelves", background_notes: "", face_visible: true, glasses: false, camera: "", notes: "",
  counts: Object.fromEntries(LETTERS.map((l) => [l, 0])), total: 0,
};
const GOOD: api.Analysis = { usable: true, reasons: [], warnings: [], metrics: {}, hand_box: [1, 2, 3, 4], crop_preview: "data:image/png;base64,AAAA" };
const BAD: api.Analysis = { usable: false, reasons: ["blurry (sharpness 40 < 70): hold still or add light"], warnings: [], metrics: {}, hand_box: null, crop_preview: null };

function stubBrowser() {
  HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined);
  HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({ drawImage: vi.fn() }) as unknown as HTMLCanvasElement["getContext"];
  HTMLCanvasElement.prototype.toBlob = vi.fn(function toBlob(cb: BlobCallback) {
    cb(new Blob(["frame"], { type: "image/jpeg" }));
  }) as unknown as HTMLCanvasElement["toBlob"];
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) },
  });
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
}

async function openSession() {
  render(<CollectPage />);
  fireEvent.change(await screen.findByLabelText(/room/i), { target: { value: "room" } });
  fireEvent.click(screen.getByRole("button", { name: "Start session" }));
  const video = (await screen.findByLabelText(/webcam preview for data collection/i)) as HTMLVideoElement;
  Object.defineProperty(video, "readyState", { value: 4, configurable: true });
  Object.defineProperty(video, "videoWidth", { value: 960, configurable: true });
  Object.defineProperty(video, "videoHeight", { value: 720, configurable: true });
  await waitFor(() => expect(screen.queryByText(/starting camera/i)).toBeNull());
}

const chooseLetter = (l: string) => fireEvent.click(screen.getByRole("button", { name: new RegExp(`^Letter ${l},`) }));

beforeEach(() => {
  vi.mocked(api.collectStatus).mockResolvedValue(STATUS);
  vi.mocked(api.listSessions).mockResolvedValue({ sessions: [], totals: STATUS.totals });
  vi.mocked(api.createSession).mockResolvedValue(SESSION);
  vi.mocked(api.analyzeFrame).mockResolvedValue(GOOD);
  vi.mocked(api.saveSample).mockImplementation(async (input) => ({
    sample: { sample_id: `${input.letter}_1_abcd`, letter: input.letter },
    session: { ...SESSION, total: 1, counts: { ...SESSION.counts, [input.letter]: 1 } },
  }));
  vi.mocked(api.deleteSample).mockResolvedValue({ deleted: "x", session: SESSION });
  vi.mocked(predictImage).mockReset();
  collectTiming.holdIntervalMs = 1;
  stubBrowser();
});
afterEach(() => vi.clearAllMocks());

describe("CollectPage", () => {
  it("explains how to enable collection when the backend has it switched off", async () => {
    vi.mocked(api.collectStatus).mockRejectedValue(new api.CollectApiError("Not found", 404));
    render(<CollectPage />);
    expect(await screen.findByText(/ASL_QUEST_DEV_COLLECT=1/)).toBeInTheDocument();
  });

  it("creates a session from the room description before anything else", async () => {
    render(<CollectPage />);
    const start = await screen.findByRole("button", { name: "Start session" });
    expect(start).toBeDisabled(); // a room name is required
    fireEvent.change(screen.getByLabelText(/room/i), { target: { value: "Bedroom shelf wall" } });
    fireEvent.click(start);
    await waitFor(() => expect(api.createSession).toHaveBeenCalledWith(expect.objectContaining({ room: "Bedroom shelf wall", lighting: "lamp", background: "shelves", face_visible: true })));
  });

  it("will not capture until a letter has been chosen", async () => {
    await openSession();
    expect(screen.getByRole("button", { name: "Choose a letter first" })).toBeDisabled();
    chooseLetter("C");
    expect(screen.getByRole("button", { name: /Capture C/ })).toBeEnabled();
  });

  it("shows the quality verdict for a bad frame and refuses to save it", async () => {
    vi.mocked(api.analyzeFrame).mockResolvedValue(BAD);
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: /Capture C/ }));
    const panel = await screen.findByTestId("review-panel");
    expect(await within(panel).findByText(/blurry/)).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: /save 0 frames/i })).toBeDisabled();
    expect(api.saveSample).not.toHaveBeenCalled();
  });

  it("saves only after explicit confirmation, with the confirmed letter, hand and distance", async () => {
    await openSession();
    chooseLetter("C");
    fireEvent.change(screen.getByLabelText("Hand used"), { target: { value: "left" } });
    fireEvent.change(screen.getByLabelText("Distance from camera"), { target: { value: "near" } });
    fireEvent.click(screen.getByRole("button", { name: /Capture C/ }));
    const save = await screen.findByRole("button", { name: /Yes, this is C/ });
    expect(api.saveSample).not.toHaveBeenCalled(); // capturing alone never saves
    fireEvent.click(save);
    await waitFor(() => expect(api.saveSample).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.saveSample).mock.calls[0][0]).toMatchObject({
      sessionId: SESSION.session_id, letter: "C", hand: "left", distance: "near", mode: "single", videoSize: "960x720",
    });
    expect(await screen.findByText(/Saved 1 frame as C in session "room" \(s20261019_120000_room\)\./)).toBeInTheDocument();
  });

  it("shows which session frames will be saved into, with its id, before anything is captured", async () => {
    await openSession();
    const banner = screen.getByRole("status", { name: "Active session" });
    expect(banner).toHaveTextContent("Saving into: room");
    expect(banner).toHaveTextContent(SESSION.session_id);
    expect(banner).toHaveTextContent(/lamp · shelves · face visible · no glasses/);
  });

  it("requires choosing a session: nothing can be captured from the chooser, and resuming shows the exact session", async () => {
    const older = { ...SESSION, session_id: "s20261019_100000_plain-wall", room: "plain-wall", total: 87, created_at: "2026-10-19T10:00:00+00:00" };
    const newer = { ...SESSION, session_id: "s20261019_110000_bedroom", room: "bedroom", total: 0, created_at: "2026-10-19T11:00:00+00:00" };
    vi.mocked(api.listSessions).mockResolvedValue({ sessions: [older, newer], totals: STATUS.totals });
    render(<CollectPage />);
    expect(await screen.findByRole("button", { name: "Start session" })).toBeInTheDocument();
    expect(screen.queryByLabelText(/webcam preview for data collection/i)).toBeNull(); // no camera, no capture yet
    expect(screen.queryByRole("status", { name: "Active session" })).toBeNull();
    // both sessions are distinguishable by id and frame count
    expect(screen.getByRole("button", { name: `Resume session ${older.session_id}` })).toHaveTextContent("87 frames");
    fireEvent.click(screen.getByRole("button", { name: `Resume session ${newer.session_id}` }));
    expect(await screen.findByRole("status", { name: "Active session" })).toHaveTextContent(newer.session_id);
  });

  it("warns when a new session would reuse an existing room name", async () => {
    vi.mocked(api.listSessions).mockResolvedValue({ sessions: [{ ...SESSION, room: "Bedroom", total: 14 }], totals: STATUS.totals });
    render(<CollectPage />);
    fireEvent.change(await screen.findByLabelText(/^Room \(e\.g\./), { target: { value: "bedroom" } });
    expect(await screen.findByRole("note")).toHaveTextContent(/already exists.*14 frames.*SEPARATE/s);
  });

  it("'Change session' returns to the chooser so the next frames cannot silently go to the old session", async () => {
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: "Change session" }));
    expect(await screen.findByRole("button", { name: "Start session" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Capture C/ })).toBeNull();
    expect(api.saveSample).not.toHaveBeenCalled();
  });

  it("rejecting a capture saves nothing", async () => {
    await openSession();
    chooseLetter("W");
    fireEvent.click(screen.getByRole("button", { name: /Capture W/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Reject/ }));
    expect(await screen.findByText(/nothing was saved/i)).toBeInTheDocument();
    expect(api.saveSample).not.toHaveBeenCalled();
  });

  it("shows the server's reason when it refuses a frame (e.g. a near-duplicate)", async () => {
    vi.mocked(api.saveSample).mockRejectedValue(new api.CollectApiError("too similar to a recent frame", 422, ["near-duplicate"]));
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: /Capture C/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Yes, this is C/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/too similar.*near-duplicate/);
  });

  it("captures a 10-frame hold sequence and saves only the usable frames under one sequence id", async () => {
    let n = 0;
    vi.mocked(api.analyzeFrame).mockImplementation(async () => (n++ % 5 === 4 ? BAD : GOOD)); // 2 of 10 unusable
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: "Capture a hold sequence of 10 frames" }));
    const save = await screen.findByRole("button", { name: /save 8 frames/i });
    fireEvent.click(save);
    await waitFor(() => expect(api.saveSample).toHaveBeenCalledTimes(8));
    const calls = vi.mocked(api.saveSample).mock.calls.map((c) => c[0]);
    expect(new Set(calls.map((c) => c.sequenceId)).size).toBe(1);
    expect(calls.every((c) => c.mode === "hold" && c.letter === "C")).toBe(true);
    expect(calls.map((c) => c.frameIndex)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it("can undo the last save", async () => {
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: /Capture C/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Yes, this is C/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Undo last save" }));
    await waitFor(() => expect(api.deleteSample).toHaveBeenCalledWith(SESSION.session_id, "C_1_abcd"));
  });

  it("never consults the recognizer for a label", async () => {
    await openSession();
    chooseLetter("C");
    fireEvent.click(screen.getByRole("button", { name: /Capture C/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Yes, this is C/ }));
    await waitFor(() => expect(api.saveSample).toHaveBeenCalled());
    expect(predictImage).not.toHaveBeenCalled();
  });
});
