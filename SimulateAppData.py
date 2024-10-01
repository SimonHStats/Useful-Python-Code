import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def generate_performance_data(n_samples=1000, base_mean=100, base_std=20, skew=5, truncate_at=None):
    # Generate base data
    base_data = np.random.normal(base_mean, base_std, n_samples)
    
    # Apply skew
    skewed_data = skewnorm.rvs(a=skew, loc=base_mean, scale=base_std, size=n_samples)
    
    # Truncate data if specified
    if truncate_at is not None:
        skewed_data = np.clip(skewed_data, 0, truncate_at)
    
    # Add some random noise
    noise = np.random.normal(0, base_std * 0.1, n_samples)
    final_data = skewed_data + noise
    
    # Ensure all values are positive
    final_data = np.maximum(final_data, 0)
    
    return final_data

def add_time_based_patterns(df, start_date, interval_minutes=5):
    # Generate timestamps
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) for i in range(len(df))]
    df['timestamp'] = timestamps
    
    # Add daily pattern (slower during business hours)
    df['hour'] = df['timestamp'].dt.hour
    df['is_business_hours'] = (df['hour'] >= 9) & (df['hour'] < 17)
    df.loc[df['is_business_hours'], 'response_time'] *= 1.2
    
    # Add weekly pattern (slower on weekdays)
    df['is_weekday'] = df['timestamp'].dt.dayofweek < 5
    df.loc[df['is_weekday'], 'response_time'] *= 1.1
    
    return df

def add_periodic_spikes(df, spike_interval_hours=6, spike_duration_minutes=15, spike_magnitude=2):
    # Add periodic spikes
    df['minutes_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    df['is_spike'] = ((df['minutes_since_start'] % (spike_interval_hours * 60)) < spike_duration_minutes)
    df.loc[df['is_spike'], 'response_time'] *= spike_magnitude
    
    return df

def add_random_degradation(df, degradation_probability=0.01, degradation_duration_hours=2, degradation_factor=1.5):
    # Add random degradations
    degradation_starts = df.sample(frac=degradation_probability).index
    for start in degradation_starts:
        end = start + int(degradation_duration_hours * 60 / 5)  # Assuming 5-minute intervals
        df.loc[start:end, 'response_time'] *= degradation_factor
    
    return df

# Generate base data
start_date = datetime(2024, 1, 1)
n_days = 7
n_samples = n_days * 24 * 12  # 7 days of data at 5-minute intervals
response_times = generate_performance_data(n_samples=n_samples, base_mean=100, base_std=20, skew=5, truncate_at=500)

# Create DataFrame and add time-based patterns
df = pd.DataFrame({'response_time': response_times})
df = add_time_based_patterns(df, start_date)
df = add_periodic_spikes(df)
df = add_random_degradation(df)

# Plot time series
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['response_time'])
plt.title('Response Times Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Response Time (ms)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(df['response_time'], bins=50, edgecolor='black')
plt.title('Distribution of Response Times')
plt.xlabel('Response Time (ms)')
plt.ylabel('Frequency')
plt.show()

# Print summary statistics
print(df.describe())
