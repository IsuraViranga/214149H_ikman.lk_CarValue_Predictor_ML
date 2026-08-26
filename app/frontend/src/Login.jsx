/**
 * Login.jsx — one form that handles both signing in and creating an account.
 */

import { useState } from "react";
import { useAuth } from "./AuthContext";

const C = {
  bg: "#0A0E1A", card: "#161D2E", border: "#1E2A3A", accent: "#00D4FF",
  red: "#FF4D6A", text: "#E8F0FE", muted: "#7A8BA0",
};

const inputStyle = {
  width: "100%", padding: "12px 14px", borderRadius: 10,
  background: "#0F1626", border: `1px solid ${C.border}`,
  color: C.text, fontSize: 14, fontFamily: "inherit", outline: "none",
};

export default function Login() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState("login");   // "login" | "register"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const isRegister = mode === "register";

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await (isRegister ? register(email, password) : login(email, password));
      // On success AuthProvider sets the user and Root swaps in the predictor.
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, display: "flex",
                  alignItems: "center", justifyContent: "center", padding: 20,
                  fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <div style={{ width: "100%", maxWidth: 400, background: C.card,
                    border: `1px solid ${C.border}`, borderRadius: 18, padding: 32 }}>

        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: C.text }}>
          CarValue<span style={{ color: C.accent }}>LK</span>
        </h1>
        <p style={{ margin: "8px 0 26px", fontSize: 13, color: C.muted }}>
          {isRegister ? "Create an account to value your vehicle."
                      : "Sign in to value your vehicle."}
        </p>

        <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <label style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
                            color: C.muted, textTransform: "uppercase" }}>Email</label>
            <input type="email" value={email} required autoComplete="email"
                   onChange={(e) => setEmail(e.target.value)}
                   style={{ ...inputStyle, marginTop: 6 }} placeholder="you@example.com" />
          </div>

          <div>
            <label style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
                            color: C.muted, textTransform: "uppercase" }}>Password</label>
            {/* type="password" keeps it off-screen; HTTPS keeps it off the wire. */}
            <input type="password" value={password} required minLength={8}
                   autoComplete={isRegister ? "new-password" : "current-password"}
                   onChange={(e) => setPassword(e.target.value)}
                   style={{ ...inputStyle, marginTop: 6 }} placeholder="At least 8 characters" />
          </div>

          {error && (
            <div style={{ background: "rgba(255,77,106,0.1)", border: `1px solid ${C.red}`,
                          borderRadius: 10, padding: "10px 12px", fontSize: 13, color: C.red }}>
              {error}
            </div>
          )}

          <button type="submit" disabled={busy}
                  style={{ padding: "13px 16px", borderRadius: 10, border: "none",
                           background: busy ? C.border : C.accent,
                           color: busy ? C.muted : "#04121A",
                           fontSize: 14, fontWeight: 800, fontFamily: "inherit",
                           cursor: busy ? "default" : "pointer" }}>
            {busy ? "Please wait…" : isRegister ? "Create account" : "Sign in"}
          </button>
        </form>

        <div style={{ marginTop: 22, textAlign: "center", fontSize: 13, color: C.muted }}>
          {isRegister ? "Already have an account?" : "No account yet?"}{" "}
          <button type="button"
                  onClick={() => { setMode(isRegister ? "login" : "register"); setError(null); }}
                  style={{ background: "none", border: "none", color: C.accent,
                           fontSize: 13, fontWeight: 700, fontFamily: "inherit",
                           cursor: "pointer", padding: 0 }}>
            {isRegister ? "Sign in" : "Create one"}
          </button>
        </div>
      </div>
    </div>
  );
}
