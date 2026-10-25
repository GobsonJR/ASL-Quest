import { useEffect, useState, type RefObject } from "react";
import Player from "@vimeo/player";

export type EmbedState = "loading" | "loaded" | "error";

/** How long a Vimeo embed may take to report ready before the fallback link is offered. Generous on purpose: a slow
 * connection is not a broken video. If the player becomes ready later the fallback goes away again. */
export const EMBED_READY_TIMEOUT_MS = 15000;

/** Tracks whether the Vimeo player inside `iframeRef` actually initialised, using Vimeo's official Player SDK.
 *
 * An iframe's own `load` event fires even when Vimeo serves its "this video does not exist" / privacy page, so it says
 * nothing about the video. The SDK's `ready()` resolves only once the real player is up, and rejects (NotFoundError,
 * PrivacyError, ...) when Vimeo reports the video unavailable. When Vimeo is blocked or unreachable neither happens, so a
 * timeout covers that case.
 *
 * Everything is scoped to the current `embedUrl`: a late answer from a previous sign's player can never change the state of
 * the current one, and listeners/timers are released on unmount or when the sign changes. The iframe should be keyed by its
 * URL so each sign gets its own element. */
export function useVimeoReady(iframeRef: RefObject<HTMLIFrameElement | null>, embedUrl: string | null): EmbedState {
  // Tagged with the URL it belongs to, so a previous sign's result is never shown for the current one.
  const [result, setResult] = useState<{ url: string; value: EmbedState } | null>(null);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!embedUrl || !iframe) return;
    let active = true;
    let player: Player | null = null;
    const settle = (next: EmbedState) => {
      if (active) setResult({ url: embedUrl, value: next });
    };
    const timer = window.setTimeout(() => settle("error"), EMBED_READY_TIMEOUT_MS);
    const onError = () => settle("error");

    try {
      player = new Player(iframe);
      player.on("error", onError);
      player.ready().then(
        () => {
          window.clearTimeout(timer);
          settle("loaded"); // also recovers from a timeout-triggered fallback if the player turned up late
        },
        () => {
          window.clearTimeout(timer);
          settle("error");
        }
      );
    } catch {
      // The SDK throws if the element is not a recognisable Vimeo embed.
      window.clearTimeout(timer);
      settle("error");
    }

    return () => {
      active = false;
      window.clearTimeout(timer);
      if (player) {
        player.off("error", onError);
        // destroy() also removes the iframe from the DOM; only call it once React has already detached the element.
        if (!iframe.isConnected) void player.destroy().catch(() => {});
      }
    };
  }, [embedUrl, iframeRef]);

  return result && result.url === embedUrl ? result.value : "loading";
}
