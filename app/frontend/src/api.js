/**
 * api.js — one place that knows about the API base URL and the JWT.
 *
 * Every authenticated request goes through apiFetch(), so the Authorization
 * header is attached in exactly one spot rather than repeated at each call.
 */

const API = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

// localStorage survives refresh and tab close. Readable by any JavaScript on
// the page, so it is only safe because this app renders no user-supplied HTML.
const TOKEN_KEY = "carvalue_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const saveToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

/**
 * Read the payload out of a JWT without contacting the server.
 * The payload is base64url-encoded, NOT encrypted -- anyone can do this.
 * We only use it for display; the server re-verifies the signature on
 * every request, so a tampered token here would simply be rejected there.
 */
export function decodeToken(token) {
  if (!token) return null;
  try {
    const part = token.split(".")[1];
    const padded = part.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
    const claims = JSON.parse(json);
    if (claims.exp && claims.exp * 1000 < Date.now()) return null; // already expired
    return { id: claims.sub, email: claims.email, role: claims.role, exp: claims.exp };
  } catch {
    return null;
  }
}

/** Fired when a request comes back 401 so the UI can drop to the login screen. */
export const LOGOUT_EVENT = "carvalue:logout";

export async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API}${path}`, { ...options, headers });

  // Token missing, expired or invalid -> stop pretending we are logged in.
  if (res.status === 401) {
    clearToken();
    window.dispatchEvent(new Event(LOGOUT_EVENT));
    throw new Error("Your session has expired. Please log in again.");
  }

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    // 403 = authenticated but not allowed. Distinct from 401 on purpose.
    throw new Error(data.error || data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export { API };
