/**
 * AuthContext.jsx — holds the logged-in user for the whole app.
 *
 * The user object is derived from the JWT itself rather than fetched, so a
 * page refresh restores the session with no network round-trip.
 */

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { apiFetch, clearToken, decodeToken, getToken, saveToken, LOGOUT_EVENT } from "./api";

const AuthContext = createContext(null);

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  // Lazy initialiser: read the stored token once, on first render.
  const [user, setUser] = useState(() => decodeToken(getToken()));

  // apiFetch fires this after any 401, so an expired token drops us to login.
  useEffect(() => {
    const onLogout = () => setUser(null);
    window.addEventListener(LOGOUT_EVENT, onLogout);
    return () => window.removeEventListener(LOGOUT_EVENT, onLogout);
  }, []);

  const authenticate = useCallback(async (path, email, password) => {
    const data = await apiFetch(path, {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    saveToken(data.access_token);
    // Trust the decoded token over the response body: it is what every
    // later request will actually be judged on.
    setUser(decodeToken(data.access_token));
    return data;
  }, []);

  const login    = useCallback((e, p) => authenticate("/auth/login", e, p),    [authenticate]);
  const register = useCallback((e, p) => authenticate("/auth/register", e, p), [authenticate]);

  const logout = useCallback(() => {
    // Nothing to call server-side: the token is stateless, so "logging out"
    // means discarding it. It stays technically valid until it expires --
    // the trade-off for not keeping a session table.
    clearToken();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, register, logout, isAdmin: user?.role === "admin" }}>
      {children}
    </AuthContext.Provider>
  );
}
