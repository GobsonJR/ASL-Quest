import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Dialog } from "./Dialog";

afterEach(() => cleanup());

describe("Dialog", () => {
  it("has an accessible name, focuses inside, closes on Escape and restores focus", () => {
    const onClose = vi.fn();
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = render(
      <Dialog label="Example dialog" onClose={onClose}>
        <button>First</button>
        <button>Last</button>
      </Dialog>
    );
    expect(screen.getByRole("dialog", { name: "Example dialog" })).toBeInTheDocument();
    expect(document.activeElement).toBe(screen.getByText("First"));

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);

    unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it("traps Tab between first and last controls", () => {
    render(
      <Dialog label="Trap" onClose={() => {}}>
        <button>First</button>
        <button>Last</button>
      </Dialog>
    );
    screen.getByText("Last").focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(document.activeElement).toBe(screen.getByText("First"));
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(screen.getByText("Last"));
  });
});
