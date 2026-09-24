/** Reads a failed fetch Response's JSON body for a backend-authored `detail`
 * message, falling back to a caller-supplied generic message when the body
 * isn't JSON or has no `detail` field. Shared by every API module so a
 * backend rejection (400/404/etc.) reaches the user with its real reason
 * instead of a generic "request failed" string. */
export async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail ?? fallback;
  } catch {
    return fallback;
  }
}
