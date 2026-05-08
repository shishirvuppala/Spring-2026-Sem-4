import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

def generate_realistic_pollutants(n_samples):
    """Realistic air pollutant distributions"""
    
    # 1. Common urban pollution factor (traffic/industry)
    urban_factor = np.random.lognormal(mean=0, sigma=0.6, size=n_samples)
    
    # 2. Weather patterns factor
    weather_factor = np.random.normal(0, 1, n_samples)
    
    # 3. Diurnal variation (time of day effect)
    diurnal = np.sin(np.linspace(0, 4*np.pi, n_samples) + np.random.normal(0, 0.5, n_samples))

    # PM2.5 (µg/m³): Primary concern, correlates with traffic
    # Realistic: 5-150 typical, occasional spikes to 300+
    PM25_base = 20 + 15*urban_factor + 5*diurnal + np.random.lognormal(3, 0.8, n_samples)
    PM25 = np.clip(PM25_base, 5, 400)
    
    # CO: Gamma (traffic peaks) [web:19]
    CO_base = 0.5 + 0.3*urban_factor + 0.1*diurnal + np.random.gamma(1, 0.2, n_samples)
    CO = np.clip(CO_base, 0.1, 8)
    
    # NO2: Weibull (rush hour spikes)
    NO2_base = 40 + 15*urban_factor - 10*diurnal + np.random.weibull(2, n_samples)*15
    NO2 = np.clip(NO2_base, 2, 200)

    # O3: FIXED - Beta distribution (daytime peaks, anticorrelated w/NO2)
    # Beta creates realistic bounded peak shape for ozone
    O3_base = 70 - 0.5*NO2 + 20*diurnal + 10*weather_factor + np.random.beta(2, 3, n_samples)*60
    O3 = np.clip(O3_base, 5, 150)
    
    # SO2: Gumbel (rare industrial spikes) [web:19]
    SO2 = 5 + np.random.gumbel(loc=0, scale=2, size=n_samples)
    SO2 = np.clip(SO2, 1, 50)
    
    # Meteorology: realistic joint distribution
    Temperature = 15 + 10*weather_factor + 5*diurnal + np.random.normal(0, 5, n_samples)
    Humidity = 50 + 20*weather_factor - 0.3*Temperature + np.random.normal(0, 10, n_samples)
    Humidity = np.clip(Humidity, 5, 95)
    
    # IRRELEVANT: WindSpeed - uniform noise (Lasso eliminates this!)
    Wind = 4 + 2*weather_factor - 0.1*urban_factor + np.random.exponential(2, n_samples)
    Wind = np.clip(Wind, 0.3, 15)

    return PM25, CO, NO2, O3, SO2, Temperature, Humidity, Wind

# Generate base data
PM25, CO, NO2, O3, SO2, Temp, Hum, Wind = generate_realistic_pollutants(n_samples)

features_raw = np.column_stack([PM25, CO, NO2, O3, SO2, Temp, Hum, Wind])
std_devs = features_raw.std(axis=0)

desired_importance = np.array([
    4.0,    # PM2.5: 4.1 in EPA formula, primary driver
    10.6,   # CO: 10.6 in EPA formula BUT measured in tiny units (ppm)
    0.39,   # NO2: 0.39 in EPA formula
    1.33,   # O3: 1.33 in EPA formula
    0.1,   # SO2: 1.33 in EPA formula
    -0.6,   # Temp: not in AQI but affects chemistry
    -0.2,   # Humidity: affects particle formation
    0.0    # Wind: dispersion reduces concentrations
])

true_coef_raw = desired_importance / std_devs

# True sparse AQI relationship (Lasso eliminates SO2 & WindSpeed!)
X = np.column_stack([PM25, CO, NO2, O3, SO2, Temp, Hum, Wind])

AQI = np.clip(X @ true_coef_raw + np.random.normal(0, 10, n_samples), 0, 500)

# Real-world outliers (2% instrument spikes)
n_outliers = int(0.02 * n_samples)
outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)

# Pollution event outliers
PM25[outlier_idx] *= 3.5
CO[outlier_idx] *= 4.0
NO2[outlier_idx] *= 2.8
AQI[outlier_idx] = np.clip(AQI[outlier_idx] * 2.0, 0, 500)

# Final dataset
df = pd.DataFrame({
    'PM25_ugm3': PM25, 'CO_ppm': CO, 'NO2_ppb': NO2, 'O3_ppb': O3, 'SO2_ppb': SO2, 
    'Temperature_C': Temp, 'Humidity': Hum, 'WindSpeed_ms': Wind
})
df['AQI'] = AQI

# Save for analysis, and save to different dec places
df.to_csv('data.csv', index=False, float_format='%.3f')

print("✅ Dataset generated: data.csv")
print("Shape:", df.shape)
print("\nDistribution stats:")
print(df[['PM25_ugm3','CO_ppm','NO2_ppb','O3_ppb','SO2_ppb']].describe())
print("\nAQI correlations:")
print(df.corr()['AQI'].sort_values(ascending=False))
