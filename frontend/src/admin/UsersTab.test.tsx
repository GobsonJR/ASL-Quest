import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsersTab } from "./UsersTab";
import * as api from "./api";

vi.mock("../auth/AuthContext", () => ({ useAuth: () => ({ user: { id: 1, username: "boss", role: "admin" } }) }));
vi.mock("./api");

const detail = (over: Partial<api.AdminUserDetail> = {}): api.AdminUserDetail => ({
  id: 2, username: "kid", email: "kid@example.com", role: "student", is_active: true, created_at: "2026-01-01T00:00:00Z",
  xp: 40, level: 2, current_streak: 1, best_streak: 3, practice_sessions: 5, accuracy: 60, achievements: 1,
  word_sessions: 0, native_sessions: 0, recent_activity: [], ...over,
});

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.listAdminUsers).mockResolvedValue({
    page: 1, page_size: 15, total: 2,
    items: [
      { id: 1, username: "boss", email: "boss@example.com", role: "admin", created_at: "", practice_sessions: 0, xp: 0, last_activity: null },
      { id: 2, username: "kid", email: "kid@example.com", role: "student", created_at: "", practice_sessions: 5, xp: 40, last_activity: null },
    ],
  });
});

describe("UsersTab", () => {
  it("only enables Delete after the exact username is typed, then calls the API", async () => {
    vi.mocked(api.getAdminUser).mockResolvedValue(detail());
    vi.mocked(api.deleteAdminUser).mockResolvedValue({ deleted: 2 });
    render(<UsersTab />);
    fireEvent.click((await screen.findAllByText("kid"))[0]);
    fireEvent.click(await screen.findByRole("button", { name: "Delete user" }));

    const confirm = screen.getByRole("button", { name: "Delete permanently" });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: /type kid to confirm/i }), { target: { value: "ki" } });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox", { name: /type kid to confirm/i }), { target: { value: "kid" } });
    expect(confirm).toBeEnabled();
    fireEvent.click(confirm);
    await waitFor(() => expect(api.deleteAdminUser).toHaveBeenCalledWith(2, "kid"));
  });

  it("disables role, deactivate and delete on your own account", async () => {
    vi.mocked(api.getAdminUser).mockResolvedValue(detail({ id: 1, username: "boss", role: "admin" }));
    render(<UsersTab />);
    fireEvent.click((await screen.findAllByText("boss"))[0]);
    expect(await screen.findByRole("button", { name: "Make student" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Deactivate" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete user" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reset progress" })).toBeEnabled();
  });

  it("shows the server's refusal inside the dialog instead of closing it", async () => {
    vi.mocked(api.getAdminUser).mockResolvedValue(detail());
    vi.mocked(api.patchAdminUser).mockRejectedValue(new Error("Nope."));
    render(<UsersTab />);
    fireEvent.click((await screen.findAllByText("kid"))[0]);
    fireEvent.click(await screen.findByRole("button", { name: "Make admin" }));
    fireEvent.click(screen.getByRole("button", { name: "Change role" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Nope.");
  });
  it("creates an administrator through the admin-only API with the role the admin picked", async () => {
    vi.mocked(api.createAdminUser).mockResolvedValue({ id: 9, username: "newadmin", email: "n@example.com", role: "admin", is_active: true });
    render(<UsersTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Add user" }));
    const create = screen.getByRole("button", { name: "Create user" });
    expect(create).toBeDisabled(); // nothing is sent until the form is valid
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: "newadmin" } });
    fireEvent.change(screen.getByLabelText(/^email/i), { target: { value: "n@example.com" } });
    fireEvent.change(screen.getByLabelText(/temporary password/i), { target: { value: "short" } });
    expect(create).toBeDisabled();
    fireEvent.change(screen.getByLabelText(/temporary password/i), { target: { value: "long-enough-1" } });
    fireEvent.click(screen.getByRole("radio", { name: /administrator/i }));
    fireEvent.click(create);
    await waitFor(() => expect(api.createAdminUser).toHaveBeenCalledWith({ username: "newadmin", email: "n@example.com", password: "long-enough-1", role: "admin" }));
    expect(await screen.findByRole("status")).toHaveTextContent(/Created administrator account for newadmin/);
  });

  it("defaults a new account to a learner and shows the server refusal (e.g. a taken name)", async () => {
    vi.mocked(api.createAdminUser).mockRejectedValue(new Error("Username or email already registered."));
    render(<UsersTab />);
    fireEvent.click(await screen.findByRole("button", { name: "Add user" }));
    expect(screen.getByRole("radio", { name: /learner/i })).toBeChecked();
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: "kid" } });
    fireEvent.change(screen.getByLabelText(/^email/i), { target: { value: "kid@example.com" } });
    fireEvent.change(screen.getByLabelText(/temporary password/i), { target: { value: "long-enough-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Create user" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/already registered/);
    expect(vi.mocked(api.createAdminUser).mock.calls[0][0].role).toBe("student");
  });
});
