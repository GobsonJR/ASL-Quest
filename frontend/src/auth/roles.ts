import type { NavPage } from "../game/constants";

/** Pages only an administrator may open. The browser check is a convenience: the backend enforces the same rule on every
 * admin and dataset-collection endpoint, so changing state in the browser can never grant access to anything. */
export const ADMIN_PAGES: readonly NavPage[] = ["admin", "mentor-dashboard"];

export const isAdminRole = (role: string | null | undefined): boolean => role === "admin";

export const isAdminPage = (page: NavPage): boolean => ADMIN_PAGES.includes(page);
