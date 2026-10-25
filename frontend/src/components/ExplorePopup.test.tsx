import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { ExplorePopup } from "./ExplorePopup";

describe("ExplorePopup", () => {
  it("shows the three learning paths with positive wording and the mastered count", () => {
    render(<ExplorePopup lettersMastered={4} onSelect={vi.fn()} onClose={vi.fn()} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /where would you like to go/i })).toBeInTheDocument();
    for (const title of ["Alphabet", "Words", "Native Signs"]) {
      expect(screen.getByText(title).closest("button")).toBeInTheDocument();
    }
    expect(screen.getByText("4 / 26 letters mastered")).toBeInTheDocument();
    expect(screen.getByRole("dialog").textContent ?? "").not.toMatch(/not supported|unavailable|coming soon|limitation/i);
  });

  it("navigates to the chosen path", () => {
    const onSelect = vi.fn();
    render(<ExplorePopup lettersMastered={0} onSelect={onSelect} onClose={vi.fn()} />);
    fireEvent.click(screen.getByText("Native Signs").closest("button")!);
    expect(onSelect).toHaveBeenCalledWith("native");
    fireEvent.click(screen.getByText("Words").closest("button")!);
    expect(onSelect).toHaveBeenCalledWith("words");
    fireEvent.click(screen.getByText("Alphabet").closest("button")!);
    expect(onSelect).toHaveBeenCalledWith("alphabet");
  });

  it("closes with the close button and with Escape", () => {
    const onClose = vi.fn();
    render(<ExplorePopup lettersMastered={0} onSelect={vi.fn()} onClose={onClose} />);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
