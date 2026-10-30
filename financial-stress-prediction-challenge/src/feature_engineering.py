"""
Feature Engineering Module for Liquidity Stress Prediction Challenge
Extracts domain-specific financial health, cash-flow, and longitudinal trajectory indicators.
"""
import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Months order: M1 (most recent) to M6 (oldest)
    months = [1, 2, 3, 4, 5, 6]
    
    new_cols = {}
    net_cf_list = []
    deficit_flags = []
    zero_bal_flags = []
    
    # Weights for linear trend slope across 6 months (from M6=oldest to M1=most recent)
    # x = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], sum(x^2) = 17.5
    trend_weights = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) / 17.5

    for m in months:
        # 1. Monthly Inflows
        inflow = (
            df[f"m{m}_received_total_value"]
            + df[f"m{m}_deposit_total_value"]
            + df[f"m{m}_transfer_from_bank_total_value"]
        )
        new_cols[f"m{m}_total_inflow"] = inflow
        
        # 2. Monthly Outflows
        outflow = (
            df[f"m{m}_withdraw_total_value"]
            + df[f"m{m}_mm_send_total_value"]
            + df[f"m{m}_paybill_total_value"]
            + df[f"m{m}_merchantpay_total_value"]
        )
        new_cols[f"m{m}_total_outflow"] = outflow
        
        # 3. Monthly Net Cash Flow & Deficit Flag
        net_cf = inflow - outflow
        new_cols[f"m{m}_net_cash_flow"] = net_cf
        net_cf_list.append(net_cf)
        
        is_deficit = (net_cf < 0).astype(np.int8)
        new_cols[f"m{m}_is_deficit"] = is_deficit
        deficit_flags.append(is_deficit)
        
        # 4. Burn Rate & Cash Reserve Days
        new_cols[f"m{m}_burn_rate"] = outflow / (inflow + 1.0)
        daily_bal = df[f"m{m}_daily_avg_bal"]
        new_cols[f"m{m}_cash_reserve_days"] = daily_bal / ((outflow / 30.0) + 1.0)
        
        # 5. Withdrawal Pressure
        new_cols[f"m{m}_withdraw_pressure"] = df[f"m{m}_withdraw_total_value"] / (outflow + 1.0)
        
        # 6. Zero Balance Indicator
        is_zero_bal = (daily_bal == 0).astype(np.int8)
        new_cols[f"m{m}_is_zero_bal"] = is_zero_bal
        zero_bal_flags.append(is_zero_bal)
        
        # 7. Inflow and Outflow Transaction Volumes
        in_vol = (
            df[f"m{m}_received_volume"]
            + df[f"m{m}_deposit_volume"]
            + df[f"m{m}_transfer_from_bank_volume"]
        )
        out_vol = (
            df[f"m{m}_withdraw_volume"]
            + df[f"m{m}_mm_send_volume"]
            + df[f"m{m}_paybill_volume"]
            + df[f"m{m}_merchantpay_volume"]
        )
        new_cols[f"m{m}_inflow_volume"] = in_vol
        new_cols[f"m{m}_outflow_volume"] = out_vol
        new_cols[f"m{m}_avg_outflow_size"] = outflow / (out_vol + 1.0)

    # --- Longitudinal / Historical Dynamics across the 6 Months ---
    
    # Balance matrix ordered from oldest (M6) to newest (M1)
    bal_matrix = df[[f"m{m}_daily_avg_bal" for m in reversed(months)]].values
    new_cols["bal_slope_6m"] = np.dot(bal_matrix, trend_weights)
    new_cols["bal_mean_6m"] = np.mean(bal_matrix, axis=1)
    new_cols["bal_std_6m"] = np.std(bal_matrix, axis=1)
    new_cols["bal_min_6m"] = np.min(bal_matrix, axis=1)
    new_cols["bal_max_6m"] = np.max(bal_matrix, axis=1)
    new_cols["bal_cv_6m"] = new_cols["bal_std_6m"] / (new_cols["bal_mean_6m"] + 1.0)
    
    # Net Cash Flow matrix ordered from M6 to M1
    net_cf_matrix = np.column_stack([net_cf_list[m-1] for m in reversed(months)])
    new_cols["net_cf_slope_6m"] = np.dot(net_cf_matrix, trend_weights)
    new_cols["net_cf_mean_6m"] = np.mean(net_cf_matrix, axis=1)
    new_cols["net_cf_std_6m"] = np.std(net_cf_matrix, axis=1)
    new_cols["net_cf_min_6m"] = np.min(net_cf_matrix, axis=1)

    # Inflows matrix ordered from M6 to M1
    inflows_matrix = np.column_stack([new_cols[f"m{m}_total_inflow"] for m in reversed(months)])
    new_cols["inflow_slope_6m"] = np.dot(inflows_matrix, trend_weights)
    new_cols["inflow_mean_6m"] = np.mean(inflows_matrix, axis=1)

    # Outflows matrix ordered from M6 to M1
    outflows_matrix = np.column_stack([new_cols[f"m{m}_total_outflow"] for m in reversed(months)])
    new_cols["outflow_slope_6m"] = np.dot(outflows_matrix, trend_weights)
    new_cols["outflow_mean_6m"] = np.mean(outflows_matrix, axis=1)

    # Cumulative Counts over 6 Months
    new_cols["months_in_deficit_6m"] = np.sum(deficit_flags, axis=0)
    new_cols["months_zero_bal_6m"] = np.sum(zero_bal_flags, axis=0)
    new_cols["recent_2m_in_deficit"] = ((new_cols["m1_is_deficit"] == 1) & (new_cols["m2_is_deficit"] == 1)).astype(np.int8)
    
    # Momentum & Deltas (M1 vs M2, M1 vs M6)
    new_cols["bal_diff_m1_m2"] = df["m1_daily_avg_bal"] - df["m2_daily_avg_bal"]
    new_cols["bal_ratio_m1_m2"] = df["m1_daily_avg_bal"] / (df["m2_daily_avg_bal"] + 1.0)
    new_cols["bal_diff_m1_m6"] = df["m1_daily_avg_bal"] - df["m6_daily_avg_bal"]
    new_cols["bal_ratio_m1_m6"] = df["m1_daily_avg_bal"] / (df["m6_daily_avg_bal"] + 1.0)

    new_cols["net_cf_diff_m1_m2"] = new_cols["m1_net_cash_flow"] - new_cols["m2_net_cash_flow"]
    new_cols["net_cf_diff_m1_m6"] = new_cols["m1_net_cash_flow"] - new_cols["m6_net_cash_flow"]
    
    new_cols["outflow_diff_m1_m2"] = new_cols["m1_total_outflow"] - new_cols["m2_total_outflow"]
    new_cols["outflow_ratio_m1_m2"] = new_cols["m1_total_outflow"] / (new_cols["m2_total_outflow"] + 1.0)
    
    new_cols["inflow_diff_m1_m2"] = new_cols["m1_total_inflow"] - new_cols["m2_total_inflow"]
    new_cols["inflow_ratio_m1_m2"] = new_cols["m1_total_inflow"] / (new_cols["m2_total_inflow"] + 1.0)

    new_cols["withdraw_diff_m1_m2"] = df["m1_withdraw_total_value"] - df["m2_withdraw_total_value"]
    new_cols["withdraw_ratio_m1_m2"] = df["m1_withdraw_total_value"] / (df["m2_withdraw_total_value"] + 1.0)

    # Demographic & Profile Interaction Ratios
    new_cols["arpu_to_bal_m1"] = df["arpu"] / (df["m1_daily_avg_bal"] + 1.0)
    new_cols["arpu_to_outflow_m1"] = df["arpu"] / (new_cols["m1_total_outflow"] + 1.0)
    new_cols["activity_weighted_bal"] = df["x_90_d_activity_rate"] * df["m1_daily_avg_bal"]
    new_cols["activity_weighted_outflow"] = df["x_90_d_activity_rate"] * new_cols["m1_total_outflow"]

    new_df = pd.DataFrame(new_cols, index=df.index)
    res = pd.concat([df, new_df], axis=1)

    # Replace potential infinite or NaN values from divisions safely
    res = res.replace([np.inf, -np.inf], np.nan)
    res = res.fillna(0)

    return res
