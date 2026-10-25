import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { AdminPage } from "./AdminPage";

let mockRole = "admin";
vi.mock("../auth/AuthContext", () => ({ useAuth: () => ({ user: { id: 1, username: "boss", role: mockRole } }) }));
vi.mock("../game/GameContext", () => ({ useGame: () => ({ navigate: vi.fn() }) }));
vi.mock("../auth/api", () => ({
  fetchAdminActivity: vi.fn().mockReturnValue(new Promise(() => {})),
  fetchAdminLetters: vi.fn().mockReturnValue(new Promise(() => {})),
  fetchAdminOverview: vi.fn().mockReturnValue(new Promise(() => {})),
  fetchAdminRecentActivity: vi.fn().mockReturnValue(new Promise(() => {})),
  fetchAdminWords: vi.fn().mockReturnValue(new Promise(() => {})),
}));
vi.mock("../admin/UsersTab", () => ({ UsersTab: () => <p>USERS TAB</p> }));
vi.mock("../admin/ContentTab", () => ({ ContentTab: () => <p>CONTENT TAB</p> }));
vi.mock("../admin/ModelsTab", () => ({ ModelsTab: () => <p>MODELS TAB</p> }));
vi.mock("../admin/AuditTab", () => ({ AuditTab: () => <p>AUDIT TAB</p> }));
vi.mock("../admin/DatasetTab", () => ({ DatasetTab: ({ onOpenCollection }: { onOpenCollection: () => void }) => <button onClick={onOpenCollection}>open collection</button> }));
vi.mock("../collect/CollectPage", () => ({ CollectPage: () => <p>COLLECTION WORKSPACE</p> }));

afterEach(() => {
  cleanup();
  mockRole = "admin";
});

describe("AdminPage", () => {
  it("gives administrators a Collect data tab that opens the dataset collection workspace", () => {
    render(<AdminPage />);
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toContain("Collect data");
    fireEvent.click(screen.getByRole("tab", { name: "Collect data" }));
    expect(screen.getByText("COLLECTION WORKSPACE")).toBeInTheDocument();
  });

  it("the Dataset tab links into the collection workspace instead of duplicating it", () => {
    render(<AdminPage />);
    fireEvent.click(screen.getByRole("tab", { name: "Dataset" }));
    fireEvent.click(screen.getByRole("button", { name: "open collection" }));
    expect(screen.getByText("COLLECTION WORKSPACE")).toBeInTheDocument();
  });

  it("shows regular users no tabs and no collection tools", () => {
    mockRole = "student";
    render(<AdminPage />);
    expect(screen.getByText("Admin access required")).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(screen.queryByText("COLLECTION WORKSPACE")).toBeNull();
  });
});
