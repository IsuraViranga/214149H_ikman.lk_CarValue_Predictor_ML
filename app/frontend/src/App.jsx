import { useState, useEffect, useCallback } from "react";

const API = "http://localhost:5000/api";

// Utility
const fmt = (n) => `Rs ${Number(n).toLocaleString("en-LK")}`;

// Design tokens
const C = {
  bg:      "#0A0E1A",
  surface: "#111827",
  card:    "#161D2E",
  border:  "#1E2A3A",
  accent:  "#00D4FF",
  accentD: "#0099BB",
  green:   "#00E5A0",
  red:     "#FF4D6A",
  amber:   "#FFB020",
  text:    "#E8F0FE",
  muted:   "#7A8BA0",
  white:   "#FFFFFF",
};

// Reusable styled components
const SelectField = ({ label, value, onChange, options, disabled }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <label style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
                    color: C.muted, textTransform: "uppercase" }}>{label}</label>
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      disabled={disabled}
      style={{
        background: C.surface, color: disabled ? C.muted : C.text,
        border: `1px solid ${C.border}`, borderRadius: 10, padding: "11px 14px",
        fontSize: 14, fontFamily: "inherit", cursor: disabled ? "not-allowed" : "pointer",
        outline: "none", appearance: "none",
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%237A8BA0' stroke-width='2'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")`,
        backgroundRepeat: "no-repeat", backgroundPosition: "right 12px center",
        paddingRight: 36, transition: "border-color 0.2s",
      }}
      onFocus={e => e.target.style.borderColor = C.accent}
      onBlur={e => e.target.style.borderColor = C.border}
    >
      {options.map(opt =>
        typeof opt === "string"
          ? <option key={opt} value={opt}>{opt}</option>
          : <option key={opt.value} value={opt.value}>{opt.label}</option>
      )}
    </select>
  </div>
);

const NumberField = ({ label, value, onChange, min, max, step = 1, suffix = "" }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <label style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
                    color: C.muted, textTransform: "uppercase" }}>{label}</label>
    <div style={{ position: "relative" }}>
      <input
        type="number" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          width: "100%", boxSizing: "border-box",
          background: C.surface, color: C.text,
          border: `1px solid ${C.border}`, borderRadius: 10,
          padding: "11px 14px", paddingRight: suffix ? 44 : 14,
          fontSize: 14, fontFamily: "inherit", outline: "none",
          transition: "border-color 0.2s",
        }}
        onFocus={e => e.target.style.borderColor = C.accent}
        onBlur={e => e.target.style.borderColor = C.border}
      />
      {suffix && (
        <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
                       fontSize: 12, color: C.muted, pointerEvents: "none" }}>{suffix}</span>
      )}
    </div>
  </div>
);

// Contribution bar
const ContribBar = ({ item, maxAbs }) => {
  const pct = maxAbs > 0 ? (Math.abs(item.value) / maxAbs) * 100 : 0;
  const isUp = item.direction === "up";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0",
                  borderBottom: `1px solid ${C.border}` }}>
      <div style={{ width: 22, textAlign: "center", fontSize: 13, flexShrink: 0,
                    color: isUp ? C.green : C.red, fontWeight: 700 }}>
        {isUp ? "▲" : "▼"}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, color: C.text, marginBottom: 4,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {item.label}
        </div>
        <div style={{ height: 6, borderRadius: 3, background: C.border, overflow: "hidden" }}>
          <div style={{
            height: "100%", width: `${pct}%`, borderRadius: 3,
            background: isUp
              ? `linear-gradient(90deg, ${C.green}88, ${C.green})`
              : `linear-gradient(90deg, ${C.red}88, ${C.red})`,
            transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
          }} />
        </div>
      </div>
      <div style={{ fontSize: 12, fontWeight: 700, flexShrink: 0, width: 90, textAlign: "right",
                    color: isUp ? C.green : C.red }}>
        {isUp ? "+" : "-"}{item.formatted}
      </div>
    </div>
  );
};

// Main App
export default function App() {
  const [options, setOptions]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [result,  setResult]      = useState(null);
  const [error,   setError]       = useState(null);
  const [animIn,  setAnimIn]      = useState(false);

  // Form state
  const [brand,        setBrand]        = useState("Toyota");
  const [carModel,     setCarModel]     = useState("Aqua");
  const [condition,    setCondition]    = useState("Used");
  const [transmission, setTransmission] = useState("Automatic");
  const [bodyType,     setBodyType]     = useState("Hatchback");
  const [fuelType,     setFuelType]     = useState("Hybrid");
  const [district,     setDistrict]     = useState("Colombo");
  const [year,         setYear]         = useState(2018);
  const [mileage,      setMileage]      = useState(50000);
  const [engineCC,     setEngineCC]     = useState(1500);
  const [hasTrim,      setHasTrim]      = useState(1);

  // Load options from API
  useEffect(() => {
    fetch(`${API}/options`)
      .then(r => r.json())
      .then(d => { setOptions(d); })
      .catch(() => setError("Cannot connect to Flask server. Make sure it's running on port 5000."));
  }, []);

  // Reset model when brand changes
  useEffect(() => {
    if (options && options.brand_models[brand]) {
      setCarModel(options.brand_models[brand][0] || "Other Model");
    }
  }, [brand, options]);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setAnimIn(false);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brand, model: carModel, condition, transmission,
          body_type: bodyType, fuel_type: fuelType, district,
          year, mileage_km: mileage, engine_cc: engineCC, has_trim: hasTrim,
        }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      setResult(data);
      setTimeout(() => setAnimIn(true), 50);
    } catch (e) {
      setError(e.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [brand, carModel, condition, transmission, bodyType, fuelType,
      district, year, mileage, engineCC, hasTrim]);

  const modelOptions = options?.brand_models?.[brand] || ["Other Model"];
  const maxAbs = result
    ? Math.max(...result.contributions.map(c => Math.abs(c.value)), 1)
    : 1;

  // Styles
  const styles = {
    root: {
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      backgroundImage: `radial-gradient(ellipse 80% 50% at 50% -10%, #00D4FF18, transparent)`,
    },
    header: {
      background: `${C.surface}CC`, backdropFilter: "blur(20px)",
      borderBottom: `1px solid ${C.border}`, padding: "18px 32px",
      display: "flex", alignItems: "center", gap: 14,
      position: "sticky", top: 0, zIndex: 100,
    },
    logo: {
      width: 38, height: 38, borderRadius: 10,
      background: `linear-gradient(135deg, ${C.accent}, ${C.accentD})`,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 18, flexShrink: 0,
    },
    main: {
      maxWidth: 1200, margin: "0 auto", padding: "32px 24px",
      display: "grid", gridTemplateColumns: "1fr 1fr",
      gap: 24, alignItems: "start",
    },
    card: {
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 18, padding: 28, overflow: "hidden",
    },
    sectionTitle: {
      fontSize: 11, fontWeight: 700, letterSpacing: "0.15em",
      color: C.accent, textTransform: "uppercase", marginBottom: 20,
      display: "flex", alignItems: "center", gap: 8,
    },
    grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 },
    grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 },
    divider: { height: 1, background: C.border, margin: "24px 0" },
    btn: {
      width: "100%", padding: "16px 24px", borderRadius: 12, border: "none",
      background: loading
        ? C.border
        : `linear-gradient(135deg, ${C.accent}, ${C.accentD})`,
      color: loading ? C.muted : C.bg, fontSize: 15, fontWeight: 700,
      fontFamily: "inherit", cursor: loading ? "not-allowed" : "pointer",
      transition: "all 0.2s", letterSpacing: "0.05em",
      display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
      marginTop: 28,
    },
    priceCard: {
      background: `linear-gradient(135deg, ${C.accent}18, ${C.green}08)`,
      border: `1px solid ${C.accent}40`, borderRadius: 16,
      padding: "28px 24px", textAlign: "center", marginBottom: 24,
      opacity: animIn ? 1 : 0,
      transform: animIn ? "translateY(0)" : "translateY(16px)",
      transition: "opacity 0.5s ease, transform 0.5s ease",
    },
    rangeRow: {
      display: "flex", justifyContent: "space-between", gap: 12, marginTop: 16,
    },
    rangeBadge: (color) => ({
      flex: 1, padding: "10px 14px", borderRadius: 10,
      background: `${color}14`, border: `1px solid ${color}40`,
      textAlign: "center",
    }),
    tagRow: {
      display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20,
      opacity: animIn ? 1 : 0,
      transition: "opacity 0.5s ease 0.1s",
    },
    tag: (color) => ({
      padding: "5px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600,
      background: `${color}18`, color: color, border: `1px solid ${color}30`,
    }),
    emptyState: {
      textAlign: "center", padding: "60px 20px",
      display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
    },
  };

  return (
    <div style={styles.root}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>🚗</div>
        <div>
          <div style={{ fontSize: 17, fontWeight: 700, color: C.white, letterSpacing: "-0.01em" }}>
            <span style={{ color: C.accent }}>ikman.lk</span> CarValue Predictor<span style={{ color: C.accent }}>LK</span>
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 1 }}>
            AI-Powered Vehicle Price Predictor · Sri Lanka
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%",
                        background: options ? C.green : C.red,
                        boxShadow: `0 0 8px ${options ? C.green : C.red}` }} />
          <span style={{ fontSize: 12, color: C.muted }}>
            {options ? "Model ready" : "Connecting..."}
          </span>
        </div>
      </header>

      {error && !options && (
        <div style={{ maxWidth: 700, margin: "32px auto", padding: "0 24px" }}>
          <div style={{ background: `${C.red}15`, border: `1px solid ${C.red}40`,
                        borderRadius: 14, padding: "20px 24px", color: C.red,
                        fontSize: 14, lineHeight: 1.6 }}>
            ⚠️ &nbsp;<strong>Connection error:</strong> {error}
          </div>
        </div>
      )}

      {options && (
        <main style={styles.main}>
          {/* ── LEFT: Input form ── */}
          <div style={styles.card}>
            <div style={styles.sectionTitle}>
              <span>⚙</span> Vehicle Details
            </div>

            {/* Brand & Model */}
            <div style={styles.grid2}>
              <SelectField label="Brand" value={brand} onChange={setBrand}
                options={options.brands} />
              <SelectField label="Model" value={carModel} onChange={setCarModel}
                options={modelOptions} />
            </div>

            <div style={styles.divider} />

            {/* Condition, Transmission, Body Type */}
            <div style={{ ...styles.sectionTitle, marginTop: 0 }}>
              <span>📋</span> Specifications
            </div>
            <div style={styles.grid3}>
              <SelectField label="Condition" value={condition} onChange={setCondition}
                options={options.conditions} />
              <SelectField label="Transmission" value={transmission} onChange={setTransmission}
                options={options.transmissions.filter(t => t !== "Other transmission")} />
              <SelectField label="Body Type" value={bodyType} onChange={setBodyType}
                options={options.body_types} />
            </div>

            <div style={{ height: 16 }} />

            <div style={styles.grid2}>
              <SelectField label="Fuel Type" value={fuelType} onChange={setFuelType}
                options={options.fuel_types.filter(f => f !== "Other fuel type")} />
              <SelectField label="District" value={district} onChange={setDistrict}
                options={options.districts} />
            </div>

            <div style={styles.divider} />

            {/* Numeric inputs */}
            <div style={{ ...styles.sectionTitle, marginTop: 0 }}>
              <span>📊</span> Performance & Age
            </div>
            <div style={styles.grid3}>
              <NumberField label="Year" value={year} onChange={setYear}
                min={1990} max={2026} />
              <SelectField label="Engine Capacity" value={engineCC}
                onChange={v => setEngineCC(Number(v))}
                options={options.engine_options} />
              <NumberField label="Mileage" value={mileage} onChange={setMileage}
                min={0} max={500000} step={1000} suffix="km" />
            </div>

            <div style={{ height: 16 }} />

            {/* Trim toggle */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                          background: C.surface, borderRadius: 10, padding: "12px 16px",
                          border: `1px solid ${C.border}` }}>
              <div>
                <div style={{ fontSize: 13, color: C.text, fontWeight: 600 }}>
                  Trim / Edition Specified
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>
                  Does this listing mention a trim level?
                </div>
              </div>
              <div
                onClick={() => setHasTrim(h => h === 1 ? 0 : 1)}
                style={{
                  width: 52, height: 28, borderRadius: 14, cursor: "pointer",
                  background: hasTrim ? C.accent : C.border,
                  position: "relative", transition: "background 0.25s", flexShrink: 0,
                }}
              >
                <div style={{
                  position: "absolute", top: 3, left: hasTrim ? 26 : 3,
                  width: 22, height: 22, borderRadius: "50%",
                  background: C.white, transition: "left 0.25s",
                  boxShadow: "0 2px 6px rgba(0,0,0,0.35)",
                }} />
              </div>
            </div>

            {/* Predict button */}
            <button style={styles.btn} onClick={handlePredict} disabled={loading}>
              {loading ? (
                <>
                  <span style={{ display: "inline-block", animation: "spin 1s linear infinite" }}>⟳</span>
                  Predicting...
                </>
              ) : (
                <> 🔮 &nbsp; Predict Price </>
              )}
            </button>

            {error && result === null && (
              <div style={{ marginTop: 14, background: `${C.red}15`, border: `1px solid ${C.red}40`,
                            borderRadius: 10, padding: "12px 16px", color: C.red, fontSize: 13 }}>
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* RIGHT: Results panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Price result */}
            {result ? (
              <>
                <div style={styles.priceCard}>
                  <div style={{ fontSize: 11, color: C.accent, fontWeight: 700,
                                letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 10 }}>
                    Estimated Market Price
                  </div>
                  <div style={{ fontSize: 38, fontWeight: 800, color: C.white,
                                letterSpacing: "-0.02em", lineHeight: 1.1 }}>
                    {fmt(result.predicted_price)}
                  </div>
                  <div style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>
                    Based on {result.inputs.brand} {result.inputs.model} ·{" "}
                    {result.inputs.year} · {result.inputs.condition}
                  </div>
                  <div style={styles.rangeRow}>
                    <div style={styles.rangeBadge(C.green)}>
                      <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>LOW ESTIMATE</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: C.green }}>
                        {fmt(result.price_low)}
                      </div>
                    </div>
                    <div style={styles.rangeBadge(C.amber)}>
                      <div style={{ fontSize: 10, color: C.muted, marginBottom: 3 }}>HIGH ESTIMATE</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: C.amber }}>
                        {fmt(result.price_high)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Input summary tags */}
                <div style={styles.tagRow}>
                  {[
                    [C.accent,     `${result.inputs.brand} ${result.inputs.model}`],
                    [C.green,      result.inputs.condition],
                    [C.amber,      `${result.inputs.age}yr old`],
                    [C.muted,      result.inputs.fuel_type],
                    [C.muted,      result.inputs.transmission],
                    [C.muted,      `${(result.inputs.mileage_km/1000).toFixed(0)}k km`],
                    [C.muted,      `${result.inputs.engine_cc} cc`],
                    [C.muted,      result.inputs.district],
                  ].map(([color, label], i) => (
                    <span key={i} style={styles.tag(color)}>{label}</span>
                  ))}
                </div>

                {/* Feature contributions */}
                <div style={styles.card}>
                  <div style={styles.sectionTitle}>
                    <span>📈</span> Why This Price? — Feature Contributions
                  </div>
                  <div style={{ fontSize: 12, color: C.muted, marginBottom: 16, lineHeight: 1.6 }}>
                    Each feature's contribution shows how much it pushed the predicted price
                    <span style={{ color: C.green }}> up ▲</span> or
                    <span style={{ color: C.red }}> down ▼</span> compared to an average car.
                  </div>
                  {result.contributions.map((item, i) => (
                    <div key={i} style={{
                      opacity: animIn ? 1 : 0,
                      transform: animIn ? "translateX(0)" : "translateX(-12px)",
                      transition: `opacity 0.4s ease ${0.1 + i * 0.06}s, transform 0.4s ease ${0.1 + i * 0.06}s`,
                    }}>
                      <ContribBar item={item} maxAbs={maxAbs} />
                    </div>
                  ))}
                  <div style={{ marginTop: 16, padding: "12px 14px",
                                background: C.surface, borderRadius: 10,
                                fontSize: 12, color: C.muted, lineHeight: 1.7 }}>
                    ℹ️ &nbsp;Contributions are computed by measuring how much each feature shifts
                    the prediction from the dataset average. Positive = increases price,
                    Negative = decreases price. Model accuracy: R² ≈ 0.92 on test data.
                  </div>
                </div>
              </>
            ) : (
              <div style={{ ...styles.card, ...styles.emptyState }}>
                <div style={{ fontSize: 56 }}>🚗</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.text }}>
                  Ready to predict
                </div>
                <div style={{ fontSize: 13, color: C.muted, maxWidth: 280, lineHeight: 1.7 }}>
                  Fill in the vehicle details on the left and click{" "}
                  <strong style={{ color: C.accent }}>Predict Price</strong> to see
                  the estimated market value and a full explanation.
                </div>
                {/* Live market stat cards */}
                <div style={{ width: "100%", maxWidth: 360, marginTop: 8 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.14em",
                                color: C.muted, textTransform: "uppercase", marginBottom: 12 }}>
                    🇱🇰 Sri Lanka Market Insights
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    {[
                      { value: "Rs 11.5M", label: "Median listing price", sub: "across all 2,609 ads", color: C.accent, icon: "💰" },
                      { value: "Toyota",   label: "Most listed brand",    sub: "916 cars · 35% of market", color: C.green, icon: "🏆" },
                      { value: "23.8%",    label: "Hybrid vehicles",      sub: "fastest growing segment", color: C.amber, icon: "⚡" },
                      { value: "7.3 yrs",  label: "Average car age",      sub: "86% are automatic", color: "#C084FC", icon: "📅" },
                    ].map(({ value, label, sub, color, icon }) => (
                      <div key={label} style={{
                        background: C.surface, borderRadius: 12, padding: "14px 14px",
                        border: `1px solid ${C.border}`, textAlign: "left",
                        position: "relative", overflow: "hidden",
                      }}>
                        <div style={{ position: "absolute", top: 0, left: 0, right: 0,
                                      height: 2, background: color, borderRadius: "12px 12px 0 0" }} />
                        <div style={{ fontSize: 18, marginBottom: 6 }}>{icon}</div>
                        <div style={{ fontSize: 16, fontWeight: 800, color: color,
                                      letterSpacing: "-0.01em", lineHeight: 1 }}>{value}</div>
                        <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginTop: 4 }}>{label}</div>
                        <div style={{ fontSize: 10, color: C.muted, marginTop: 2, lineHeight: 1.4 }}>{sub}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${C.bg}; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.4; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 860px) {
          main { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
