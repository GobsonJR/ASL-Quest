import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Not using vitest's `globals: true`, so RTL's automatic per-test cleanup
// (which relies on detecting a global afterEach) doesn't kick in on its own --
// register it explicitly so DOM from one test never leaks into the next.
afterEach(() => {
  cleanup();
});

// jsdom doesn't implement scrollIntoView at all -- any component that calls it
// (e.g. auto-scrolling a chat log to the latest message) throws in every test
// that renders it, unrelated to what that test is actually checking.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
