import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { AlphabetPage } from "./AlphabetPage";
import { createInitialState } from "../game/storage";
import { LETTERS } from "../game/constants";
import type { GameState } from "../game/types";

const startPractice = vi.fn();
let mockState: GameState = createInitialState();

vi.mock("../game/GameContext", () => ({
  useGame: () => ({ state: mockState, startPractice }),
}));

beforeEach(() => {
  startPractice.mockReset();
  mockState = createInitialState();
});
afterEach(() => cleanup());

describe("AlphabetPage", () => {
  it("renders all 26 letters", () => {
    render(<AlphabetPage />);
    const grid = screen.getByTestId("alphabet-grid");
    expect(within(grid).getAllByRole("button")).toHaveLength(LETTERS.length);
  });

  it("starts on the current letter (C) and launches camera practice for it", () => {
    render(<AlphabetPage />);
    fireEvent.click(screen.getByRole("button", { name: /practice c with camera/i }));
    expect(startPractice).toHaveBeenCalledWith("C");
  });

  it("locked letters explain why and offer no practice action", () => {
    render(<AlphabetPage />);
    fireEvent.click(screen.getByRole("button", { name: /letter z, locked/i }));
    expect(screen.queryByRole("button", { name: /practice z/i })).toBeNull();
    expect(screen.getByTestId("alphabet-detail")).toHaveTextContent(/beginner signs/i);
  });

  it("filters to completed letters and shows an empty state when none", () => {
    render(<AlphabetPage />);
    fireEvent.click(screen.getByRole("button", { name: "Completed" }));
    expect(screen.getByText(/no letters match/i)).toBeInTheDocument();
  });
});
