# Gold-Standard Specimen Report – Addendum

This addendum integrates OCO methodology and swing protocols into the daily Gold-Standard reporting format.

---

## 4. Standardized OCO Trade Setup (Default)

* OCO orders must be linked directly to the **AI Data Block (VWAP, R1, S1)**.  
* Example Playbook (Bearish Bias):  
  - **Entry:** Break below VWAP (`23,166`).  
  - **Stop:** Above R1 (`23,671`).  
  - **Target:** S1 (`23,501`).  

**Confidence-Based Target Adjustment:**  
- If conviction ≤ 50%, the profit target may be adjusted to:  
  - Mid-zone between VWAP and R1/S1.  
  - OR an acceptable portfolio loss threshold (X%).  

This ensures trades are still structured while acknowledging uncertainty.

---

## 5. Swing Position Review (When Active)

* **Ava:** Provides swing position snapshot (size + P/L) in addition to daily workflow.  
* **Report includes Swing Assessment Block:**  
  - Market Condition: Bullish, Bearish, Neutral/Chop.  
  - Options & Sentiment Context: Dealer positioning, skew, hedging.  
  - Macro Overlay: News, catalysts, positioning risk.  
  - **Recommendation:**  
    - ✅ Add to Position (trend reinforced, momentum intact, using OCO method).  
    - ❌ Exit Trade (bias invalidated, risk outweighs reward, exit via stop loss or order execution).  
    - ➖ Maintain (position justified, no adjustment needed).

---

## Execution Flow (Swing Positions)

* Swing trades are reviewed:  
  - **Pre-Market** – Establish positioning ahead of session.  
  - **Mid-Day (Volatility Present)** – Adjust if market drivers shift intraday.  
  - **Post-Market** – Assess close relative to levels for swing continuation.  

* Stops/targets migrate from **AI Data Block intraday levels → higher timeframe references (daily/weekly EMA, channel, or pivot zones)**.

---

## Result

Daily reports now deliver **directly tradable OCO setups** plus **swing guidance blocks**, ensuring continuity between intraday tactics and intra-week/monthly conviction trades.
