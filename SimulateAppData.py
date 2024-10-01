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

# Generate data
response_times = generate_performance_data(n_samples=10000, base_mean=100, base_std=20, skew=5, truncate_at=500)

# Create a DataFrame
df = pd.DataFrame({'response_time': response_times})

# Basic statistics
print(df.describe())

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df['response_time'], bins=50, edgecolor='black')
plt.title('Distribution of Response Times')
plt.xlabel('Response Time (ms)')
plt.ylabel('Frequency')
plt.show()
