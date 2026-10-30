# Data Exploration & Domain Architecture

## 1. Dataset Dimensions & Integrity
- **Train:** 40,000 observations × 184 columns.
- **Test:** 30,000 observations × 183 columns.
- **Completeness:** 0 missing values across all 40,000 train rows and 30,000 test rows. Clean tabular data.
- **Target Distribution:**
  - `0` (No liquidity stress): 34,000 (85.00%)
  - `1` (Liquidity stress in next 30 days): 6,000 (15.00%)
  - Ratio: 5.67 : 1 negative to positive.

## 2. Feature Hierarchy & Structure

### A. Customer Profile & Demographics (7 variables)
- `ID`: Snapshot identifier `ID_XXXXXXXXXX`
- `arpu`: Average revenue per user over the mobile money observation window
- `age`: Customer age in years
- `gender`: Categorical (M / F / other)
- `region`: Geographic zone / administrative area
- `smartphone`: Categorical indicator / device flag
- `segment`: Business segment categorization
- `earning_pattern`: Regularity / type of customer earnings

### B. Activity Variable (1 variable)
- `x_90_d_activity_rate`: Proportion of active mobile money days over the last 90 days (0.0 to 1.0).

### C. Monthly Longitudinal Behavioral Variables (29 features × 6 months = 174 variables)
Observed from `M1` (most recent month) down to `M6` (oldest month):
1. **Paybill payments**: `volume`, `total_value`, `highest_amount`, `companies`
2. **Merchant payments**: `volume`, `total_value`, `highest_amount`, `merchants`
3. **Transfers from bank**: `volume`, `total_value`, `highest_amount`, `banks`
4. **Mobile Money sends (P2P)**: `volume`, `total_value`, `highest_amount`, `recipients`
5. **Mobile Money received**: `volume`, `total_value`, `highest_amount`, `senders`
6. **Cash deposits (Agents)**: `volume`, `total_value`, `highest_amount`, `agents`
7. **Cash withdrawals (Agents)**: `volume`, `total_value`, `highest_amount`, `agents`
8. **Balance**: `daily_avg_bal` (daily average balance for that month)

## 3. High-Leverage Feature Engineering Hypotheses
1. **Cash Flow Dynamics (Inflow vs Outflow)**:
   - Inflows = `received_total_value` + `deposit_total_value` + `transfer_from_bank_total_value`
   - Outflows = `withdraw_total_value` + `mm_send_total_value` + `paybill_total_value` + `merchantpay_total_value`
   - Net Cash Flow = Inflows - Outflows per month.
   - Cash Flow Deficit: Flag if Net Cash Flow < 0 in M1 or M2.
2. **Balance Trajectory & Depletion Velocity**:
   - Relative change in balance: `(m1_daily_avg_bal - m2_daily_avg_bal) / (m2_daily_avg_bal + 1)`
   - 6-month balance trend / slope: linear regression slope across `[m6_bal, m5_bal, m4_bal, m3_bal, m2_bal, m1_bal]`.
   - Days of reserve: `m1_daily_avg_bal / (m1_outflows / 30 + 1e-5)`.
3. **Emergency Borrowing / Withdrawal Signals**:
   - Spike in withdrawal ratio: `m1_withdraw_total_value / (mean(m2..m6_withdraw_total_value) + 1)`
   - Decline in deposit frequency or sudden drop in inbound transfers.
4. **Transaction Diversity Ratios**:
   - Average transaction size: `total_value / (volume + 1e-5)`
   - Concentration risk: `highest_amount / (total_value + 1e-5)`
