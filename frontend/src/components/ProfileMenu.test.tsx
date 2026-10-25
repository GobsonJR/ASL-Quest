import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { ProfileMenu } from "./ProfileMenu";

afterEach(() => cleanup());

function renderMenu(overrides: Partial<React.ComponentProps<typeof ProfileMenu>> = {}) {
  const onNavigate = vi.fn();
  const onLogout = vi.fn();
  render(
    <div>
      <button type="button">outside</button>
      <ProfileMenu name="Sam" email="sam@example.com" isAdmin={false} currentPage="learn" onNavigate={onNavigate} onLogout={onLogout} {...overrides} />
    </div>
  );
  return { onNavigate, onLogout };
}

const trigger = () => screen.getByRole("button", { name: /account menu for sam/i });

describe("ProfileMenu", () => {
  it("shows the person's initial on a closed menu and opens a labelled menu with Profile, Settings and Log out", () => {
    renderMenu();
    expect(trigger()).toHaveTextContent("S");
    expect(trigger()).toHaveAttribute("aria-haspopup", "menu");
    expect(trigger()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("menu")).toBeNull();

    fireEvent.click(trigger());
    const menu = screen.getByRole("menu", { name: "Account" });
    expect(trigger()).toHaveAttribute("aria-expanded", "true");
    const labels = within(menu).getAllByRole("menuitem").map((item) => (item.textContent ?? "").replace(/[^A-Za-z ]/g, "").trim());
    expect(labels).toEqual(["Profile", "Settings", "Log out"]);
    expect(within(menu).getByText("sam@example.com")).toBeInTheDocument();
    expect(within(menu).getByText("Learner")).toBeInTheDocument();
  });

  it("offers the Admin workspace only to administrators", () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(screen.queryByRole("menuitem", { name: /admin workspace/i })).toBeNull();
    cleanup();

    const { onNavigate } = renderMenu({ isAdmin: true });
    fireEvent.click(trigger());
    expect(screen.getByText("Administrator")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: /admin workspace/i }));
    expect(onNavigate).toHaveBeenCalledWith("admin");
  });

  it("navigates to the existing Profile and Settings pages and closes", () => {
    const { onNavigate } = renderMenu();
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole("menuitem", { name: /profile/i }));
    expect(onNavigate).toHaveBeenLastCalledWith("profile");
    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole("menuitem", { name: /settings/i }));
    expect(onNavigate).toHaveBeenLastCalledWith("settings");
  });

  it("calls the existing logout handler, and omits Log out when there is no session to end (demo mode)", () => {
    const { onLogout } = renderMenu();
    fireEvent.click(trigger());
    fireEvent.click(screen.getByRole("menuitem", { name: /log out/i }));
    expect(onLogout).toHaveBeenCalledTimes(1);
    cleanup();

    renderMenu({ onLogout: undefined });
    fireEvent.click(trigger());
    expect(screen.queryByRole("menuitem", { name: /log out/i })).toBeNull();
  });

  it("closes with Escape and returns focus to the avatar", () => {
    renderMenu();
    fireEvent.click(trigger());
    expect(document.activeElement).toBe(screen.getAllByRole("menuitem")[0]); // keyboard users land inside the menu
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    expect(document.activeElement).toBe(trigger());
  });

  it("closes when clicking outside", () => {
    renderMenu();
    fireEvent.click(trigger());
    fireEvent.mouseDown(screen.getByRole("button", { name: "outside" }));
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("moves between items with the arrow keys, wrapping around", () => {
    renderMenu();
    fireEvent.click(trigger());
    const items = screen.getAllByRole("menuitem");
    fireEvent.keyDown(items[0], { key: "ArrowDown" });
    expect(document.activeElement).toBe(items[1]);
    fireEvent.keyDown(items[1], { key: "ArrowUp" });
    fireEvent.keyDown(items[0], { key: "ArrowUp" });
    expect(document.activeElement).toBe(items[items.length - 1]);
    fireEvent.keyDown(items[items.length - 1], { key: "Home" });
    expect(document.activeElement).toBe(items[0]);
  });
});
