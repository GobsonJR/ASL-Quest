import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AlphabetReferencePanel } from "./AlphabetReferencePanel";

describe("AlphabetReferencePanel", () => {
  it("shows the target letter and its reference image with an accessible alt", () => {
    render(<AlphabetReferencePanel letter="a" statusLabel="Show your hand" />);
    expect(screen.getByText("A")).toBeInTheDocument();
    const img = screen.getByRole("img", { name: /handshape for the letter a/i });
    expect(img).toHaveAttribute("src", "/asl-reference/a.svg");
  });

  it("swaps the reference image and letter when the target letter changes", () => {
    const { rerender } = render(<AlphabetReferencePanel letter="a" />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("/asl-reference/a.svg");

    rerender(<AlphabetReferencePanel letter="b" />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("/asl-reference/b.svg");
    expect(screen.getByRole("img")).toHaveAttribute("alt", expect.stringContaining("letter B"));
    expect(screen.getByText("B")).toBeInTheDocument();
  });

  it("surfaces a license note for the bundled reference art", () => {
    render(<AlphabetReferencePanel letter="a" />);
    expect(screen.getByText(/public-domain/i)).toBeInTheDocument();
  });
});
