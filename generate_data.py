import pandas as pd
import numpy as np
import os

# --- Configuration ---
original_file = 'medical_equipment_real_named_data.csv'
output_filename = 'augmented_medical_data_100k.csv'
N_TOTAL = 100000
N_BREAKDOWNS_TARGET = 5000 # We'll aim for a 5% breakdown rate
# ---------------------

print(f"Loading original data from '{original_file}'...")

if not os.path.exists(original_file):
    print(f"Error: Original file '{original_file}' not found.")
    print("Please make sure it's in the same directory as this script.")
else:
    df = pd.read_csv(original_file)

    # Drop the 'date' column as it's not used in the model
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Separate healthy vs breakdown
    df_healthy = df[df['breakdown_flag'] == 0].copy()
    df_breakdown = df[df['breakdown_flag'] == 1].copy()

    n_original_breakdowns = len(df_breakdown)
    print(f"Found {len(df_healthy)} healthy rows and {n_original_breakdowns} breakdown rows.")

    if n_original_breakdowns == 0:
        print("Warning: No breakdown rows (breakdown_flag=1) found in the original file!")
        print("Cannot generate new breakdown data. The output file will only have healthy data.")
        CAN_GENERATE_BREAKDOWNS = False
        n_new_breakdowns = 0
    else:
        CAN_GENERATE_BREAKDOWNS = True
        n_new_breakdowns = N_BREAKDOWNS_TARGET - n_original_breakdowns
        print(f"Will generate {n_new_breakdowns} new breakdown rows.")

    # Calculate how many new healthy rows we need
    n_original_healthy = len(df_healthy)
    n_new_healthy = N_TOTAL - n_original_healthy - n_original_breakdowns - n_new_breakdowns
    print(f"Will generate {n_new_healthy} new healthy rows.")

    # --- Generation Parameters ---

    # 1. Parameters for healthy data
    healthy_stats = df_healthy.describe()
    usage_mean_h = healthy_stats.loc['mean', 'usage_hours']
    usage_std_h = healthy_stats.loc['std', 'usage_hours']
    temp_mean_h = healthy_stats.loc['mean', 'temperature']
    temp_std_h = healthy_stats.loc['std', 'temperature']
    error_probs_h = df_healthy['error_count'].value_counts(normalize=True)
    healthy_devices_probs = df_healthy['device_name'].value_counts(normalize=True)

    # 2. Parameters for breakdown data
    if CAN_GENERATE_BREAKDOWNS:
        breakdown_stats = df_breakdown.describe()
        # Add some noise/variance, especially if std is 0 (which happens if n=1)
        usage_mean_b = breakdown_stats.loc['mean', 'usage_hours']
        usage_std_b = max(breakdown_stats.loc['std', 'usage_hours'], 1.0)
        temp_mean_b = breakdown_stats.loc['mean', 'temperature']
        temp_std_b = max(breakdown_stats.loc['std', 'temperature'], 1.0)
        error_mean_b = breakdown_stats.loc['mean', 'error_count']
        error_std_b = max(breakdown_stats.loc['std', 'error_count'], 0.5)
        breakdown_devices_probs = df_breakdown['device_name'].value_counts(normalize=True)

    # --- Generate New Data ---

    # 1. Generate new HEALTHY data
    new_healthy_data = {
        'device_name': np.random.choice(healthy_devices_probs.index, size=n_new_healthy, p=healthy_devices_probs.values),
        'usage_hours': np.clip(np.random.normal(usage_mean_h, usage_std_h, n_new_healthy), a_min=0, a_max=None),
        'temperature': np.random.normal(temp_mean_h, temp_std_h, n_new_healthy),
        'error_count': np.random.choice(error_probs_h.index, size=n_new_healthy, p=error_probs_h.values),
        'breakdown_flag': np.zeros(n_new_healthy, dtype=int)
    }
    new_healthy_df = pd.DataFrame(new_healthy_data)
    print("Generated new healthy data.")

    # 2. Generate new BREAKDOWN data (if possible)
    if CAN_GENERATE_BREAKDOWNS:
        new_breakdown_data = {
            'device_name': np.random.choice(breakdown_devices_probs.index, size=n_new_breakdowns, p=breakdown_devices_probs.values),
            'usage_hours': np.clip(np.random.normal(usage_mean_b, usage_std_b, n_new_breakdowns), a_min=0, a_max=None),
            'temperature': np.random.normal(temp_mean_b, temp_std_b, n_new_breakdowns),
            'error_count': np.clip(np.round(np.random.normal(error_mean_b, error_std_b, n_new_breakdowns)), a_min=0, a_max=None).astype(int),
            'breakdown_flag': np.ones(n_new_breakdowns, dtype=int)
        }
        new_breakdown_df = pd.DataFrame(new_breakdown_data)
        print("Generated new breakdown data.")
    else:
        new_breakdown_df = pd.DataFrame() # Empty dataframe

    # --- Combine and Save ---
    final_df = pd.concat([df, new_healthy_df, new_breakdown_df], ignore_index=True)
    
    # Shuffle the dataframe
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(output_filename, index=False)

    print("\n--- Success! ---")
    print(f"Saved {len(final_df)} rows to '{output_filename}'.")
    print("New Breakdown Flag Distribution:")
    print(final_df['breakdown_flag'].value_counts())