import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { AppShell } from "./AppShell";

const navigate = vi.fn();
const startPractice = vi.fn();
const logout = vi.fn();

let mockPage = "learn";
let mockRole: "student" | "admin" = "student";

vi.mock("../game/GameContext", () => ({
  useGame: () => ({
    page: mockPage,
    navigate,
    startPractice,
    practiceKind: "letter",
    level: 2,
    levelProgress: { current: 250, next: 1000, percent: 25 },
    state: {
      xp: 1250,
      streak: 4,
      unlockedBadges: ["first_sign", "streak_7"],
      dailyChallenge: { progress: 3, target: 10, completed: false },
    },
  }),
}));

vi.mock("../auth/AuthContext", () => ({
  useAuth: () => ({ user: { username: "sam", email: "sam@example.com", role: mockRole }, logout, syncStatus: "synced" }),
}));

function renderShell() {
  return render(
    <AppShell ready backendNote="Ready">
      <p>content</p>
    </AppShell>
  );
}

const PRIMARY = ["Learn", "Practice", "Alphabet", "Words", "Badges"];

beforeEach(() => {
  mockPage = "learn";
  mockRole = "student";
  navigate.mockReset();
  startPractice.mockReset();
});
afterEach(() => cleanup());

describe("AppShell navigation", () => {
  it("offers the five primary learning destinations on desktop and mobile", () => {
    renderShell();
    const desktop = screen.getByRole("navigation", { name: "Main navigation" });
    const mobile = screen.getByRole("navigation", { name: "Mobile navigation" });
    for (const label of PRIMARY) {
      const desktopLabel = label === "Badges" ? "Achievements" : label;
      expect(within(desktop).getByRole("button", { name: new RegExp(desktopLabel, "i") })).toBeInTheDocument();
      expect(within(mobile).getByRole("button", { name: new RegExp(label, "i") })).toBeInTheDocument();
    }
    expect(within(mobile).getAllByRole("button")).toHaveLength(5);
  });

  it("marks only the current destination with aria-current", () => {
    mockPage = "alphabet";
    renderShell();
    const mobile = screen.getByRole("navigation", { name: "Mobile navigation" });
    const current = within(mobile).getAllByRole("button").filter((b) => b.getAttribute("aria-current") === "page");
    expect(current).toHaveLength(1);
    expect(current[0]).toHaveTextContent("Alphabet");
  });

  it("Practice starts free practice instead of just switching page", () => {
    renderShell();
    const mobile = screen.getByRole("navigation", { name: "Mobile navigation" });
    fireEvent.click(within(mobile).getByRole("button", { name: /practice/i }));
    expect(startPractice).toHaveBeenCalledTimes(1);
    expect(navigate).not.toHaveBeenCalled();
  });

  it("opens and closes the mobile More menu, closes on Escape, and navigates from it", () => {
    renderShell();
    const toggle = screen.getByRole("button", { name: "More pages" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("mobile-more-menu")).not.toBeNull();

    fireEvent.keyDown(window, { key: "Escape" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("mobile-more-menu")).toBeNull();

    fireEvent.click(toggle);
    const menu = document.getElementById("mobile-more-menu") as HTMLElement;
    fireEvent.click(within(menu).getByRole("button", { name: /challenges/i }));
    expect(navigate).toHaveBeenCalledWith("challenges");
  });

  it("closes the More menu when the page changes", () => {
    const { rerender } = renderShell();
    fireEvent.click(screen.getByRole("button", { name: "More pages" }));
    expect(document.getElementById("mobile-more-menu")).not.toBeNull();
    mockPage = "challenges";
    rerender(
      <AppShell ready backendNote="Ready">
        <p>content</p>
      </AppShell>
    );
    expect(document.getElementById("mobile-more-menu")).toBeNull();
  });

  it("hides Admin and Mentor links for regular users and shows them for admins", () => {
    renderShell();
    fireEvent.click(screen.getByRole("button", { name: "More pages" }));
    expect(screen.queryByRole("button", { name: /^admin$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /mentor/i })).toBeNull();
    cleanup();

    mockRole = "admin";
    renderShell();
    fireEvent.click(screen.getByRole("button", { name: "More pages" }));
    expect(screen.getAllByRole("button", { name: /admin/i }).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("button", { name: /mentor/i }).length).toBeGreaterThan(0);
  });

  it("progress panel shows real values and is hidden below the wide breakpoint", () => {
    renderShell();
    const panel = screen.getByRole("complementary", { name: "Your progress", hidden: true });
    expect(panel).toHaveTextContent("1250");
    expect(panel).toHaveTextContent("3 / 10 signs");
    expect(panel).toHaveTextContent("2 badges earned");
    expect(panel).toHaveTextContent("Level 2");
    expect(panel.className).toContain("hidden");
    expect(panel.className).toContain("min-[1500px]:block");
  });
});

describe("AppShell account menu (top-right)", () => {
  it("puts the avatar and its dropdown in the header, with Profile, Settings and Log out", () => {
    renderShell();
    const header = screen.getByRole("banner");
    const avatar = within(header).getByRole("button", { name: /account menu for sam/i });
    fireEvent.click(avatar);
    fireEvent.click(screen.getByRole("menuitem", { name: /log out/i }));
    expect(logout).toHaveBeenCalledTimes(1);
  });

  it("opens the existing Profile and Settings pages from the menu", () => {
    renderShell();
    fireEvent.click(screen.getByRole("button", { name: /account menu for sam/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /profile/i }));
    expect(navigate).toHaveBeenLastCalledWith("profile");
    fireEvent.click(screen.getByRole("button", { name: /account menu for sam/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /settings/i }));
    expect(navigate).toHaveBeenLastCalledWith("settings");
  });

  it("no longer duplicates Profile, Settings or Log out in the sidebar, bottom bar or More menu", () => {
    renderShell();
    const desktop = screen.getByRole("navigation", { name: "Main navigation" });
    const mobile = screen.getByRole("navigation", { name: "Mobile navigation" });
    for (const nav of [desktop, mobile]) {
      expect(within(nav).queryByRole("button", { name: /profile|settings|log out/i })).toBeNull();
    }
    fireEvent.click(screen.getByRole("button", { name: "More pages" }));
    const more = document.getElementById("mobile-more-menu") as HTMLElement;
    expect(within(more).queryByRole("button", { name: /profile|settings|log out|collect/i })).toBeNull();
  });

  it("shows the Admin workspace entry in the menu for administrators only", () => {
    renderShell();
    fireEvent.click(screen.getByRole("button", { name: /account menu for sam/i }));
    expect(screen.queryByRole("menuitem", { name: /admin workspace/i })).toBeNull();
    cleanup();
    mockRole = "admin";
    renderShell();
    fireEvent.click(screen.getByRole("button", { name: /account menu for sam/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /admin workspace/i }));
    expect(navigate).toHaveBeenLastCalledWith("admin");
  });
});
