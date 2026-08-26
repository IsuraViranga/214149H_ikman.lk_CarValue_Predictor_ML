/**
 * Root.jsx — decides whether to show the login screen or the predictor.
 *
 * Kept separate from App.jsx so the existing prediction UI needed almost no
 * changes: it is simply rendered once a token exists.
 */

import { AuthProvider, useAuth } from "./AuthContext";
import Login from "./Login";
import App from "./App";

const C = {
  card: "#161D2E", border: "#1E2A3A", accent: "#00D4FF",
  amber: "#FFB020", text: "#E8F0FE", muted: "#7A8BA0",
};

function UserBar() {
  const { user, logout, isAdmin } = useAuth();
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end",
                  gap: 12, padding: "10px 20px", background: C.card,
                  borderBottom: `1px solid ${C.border}`,
                  fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <span style={{ fontSize: 13, color: C.muted }}>{user.email}</span>

      {/* Role comes from the JWT claim, not a lookup. */}
      <span style={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.1em",
                     textTransform: "uppercase", padding: "3px 8px", borderRadius: 6,
                     color: isAdmin ? "#04121A" : C.accent,
                     background: isAdmin ? C.amber : "rgba(0,212,255,0.12)" }}>
        {user.role}
      </span>

      <button onClick={logout}
              style={{ background: "none", border: `1px solid ${C.border}`,
                       borderRadius: 8, padding: "6px 12px", color: C.text,
                       fontSize: 12, fontWeight: 700, fontFamily: "inherit",
                       cursor: "pointer" }}>
        Sign out
      </button>
    </div>
  );
}

function Gate() {
  const { user } = useAuth();
  if (!user) return <Login />;
  return (
    <>
      <UserBar />
      <App />
    </>
  );
}

export default function Root() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
