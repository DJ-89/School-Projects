import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('/workspace/phivolcs_earthquake_data.csv')

# Preprocess data
df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')
cutoff_date = pd.to_datetime('2025-10-15')
df = df[df['Date_Time_PH'] <= cutoff_date]

# Convert Depth_In_Km to numeric, handling any non-numeric values
df['Depth_In_Km'] = pd.to_numeric(df['Depth_In_Km'], errors='coerce')

# Drop rows with missing coordinates, magnitude, or depth
df = df.dropna(subset=['Latitude', 'Longitude', 'Magnitude', 'Depth_In_Km'])

# Create binary target variable: is_significant (Magnitude >= 4.0)
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# Parse General_Location into broader regions (Luzon, Visayas, Mindanao)
def classify_region(location):
    if pd.isna(location):
        return 'Unknown'
    
    location_lower = location.lower()
    
    # Luzon regions
    luzon_regions = ['bulacan', 'pampanga', 'bataan', 'angeles', 'baguio', 'benguet', 'tarlac', 
                    'nueva ecija', 'zambales', 'pangasinan', 'la union', 'ilocos', 'abracan',
                    'abra', 'apayao', 'benguet', 'ifugao', 'kalinga', 'mountain province',
                    'cagayan', 'isabela', 'nueva vizcaya', 'quirino', 'batanes', 'basco',
                    'rizon', 'marinduque', 'romblon', 'quezon', 'bicol', 'albay', 'camarines',
                    'catanduanes', 'masbate', 'sorsogon', 'aurora', 'bataan', 'bulacan',
                    'nueva ecija', 'pampanga', 'tarlac', 'zambales', 'manila', 'quezon city',
                    'makati', 'pasig', 'mandaluyong', 'san juan', 'caloocan', 'malabon',
                    'navotas', 'valenzuela', 'marikina', 'pasay', 'paranaque', 'muntinlupa',
                    'las pinas', 'makati', 'taguig', 'pateros', 'cavite', 'laguna', 'batangas',
                    'rizal', 'cavite', 'laguna', 'batangas', 'quezon', 'marinduque']
    
    # Visayas regions
    visayas_regions = ['cebu', 'cebu city', 'negros', 'negros oriental', 'negros occidental',
                      'bohol', 'siquijor', 'biliran', 'leyte', 'southern leyte', 'western samar',
                      'eastern samar', 'northern samar', 'samara', 'iloilo', 'capiz', 'antique',
                      'guimaras', 'aclar', 'palawan', 'romblon', 'mindoro', 'marinduque',
                      'panay', 'masbate', 'catanduanes', 'sorsogon']
    
    # Mindanao regions
    mindanao_regions = ['davao', 'davao del sur', 'davao del norte', 'davao oriental', 'davao occidental',
                       'cotabato', 'north cotabato', 'south cotabato', 'sultan kudarat', 'sarangani',
                       'general santos', 'butuan', 'surigao', 'surigao del norte', 'surigao del sur',
                       'agusan', 'agusan del norte', 'agusan del sur', 'misamis', 'misamis oriental',
                       'misamis occidental', 'camiguin', 'zamboanga', 'zamboanga del norte',
                       'zamboanga del sur', 'zamboanga sibugay', 'lanao', 'lanao del norte',
                       'lanao del sur', 'maguindanao', 'cagayan de oro', 'iligan', 'marawi',
                       'malaybalay', 'valencia', 'dipolog', 'roxas', 'bukidnon', 'camiguin',
                       'misamis', 'tawi-tawi', 'sulu', 'basilan']
    
    for region in luzon_regions:
        if region in location_lower:
            return 'Luzon'
    
    for region in visayas_regions:
        if region in location_lower:
            return 'Visayas'
    
    for region in mindanao_regions:
        if region in location_lower:
            return 'Mindanao'
    
    return 'Unknown'

df['Region'] = df['General_Location'].apply(classify_region)

# Create a sample for visualization to avoid memory issues
sample_size = min(5000, len(df))
df_viz = df.sample(n=sample_size, random_state=42)

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Distribution of earthquake magnitudes
plt.subplot(2, 3, 1)
plt.hist(df_viz['Magnitude'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')

# Plot 2: Earthquakes by region
plt.subplot(2, 3, 2)
region_counts = df_viz['Region'].value_counts()
plt.bar(region_counts.index, region_counts.values)
plt.title('Earthquakes by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Plot 3: Significant vs non-significant earthquakes by region
plt.subplot(2, 3, 3)
region_sig = df_viz.groupby(['Region', 'is_significant']).size().unstack(fill_value=0)
region_sig.plot(kind='bar', ax=plt.gca(), stacked=True)
plt.title('Earthquakes by Region and Significance')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(['Non-Significant', 'Significant'])
plt.xticks(rotation=45)

# Plot 4: Geographic distribution
plt.subplot(2, 3, 4)
scatter = plt.scatter(df_viz['Longitude'], df_viz['Latitude'], c=df_viz['Magnitude'], 
                     cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter)
plt.title('Geographic Distribution of Earthquakes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Plot 5: Depth vs Magnitude
plt.subplot(2, 3, 5)
plt.scatter(df_viz['Depth_In_Km'], df_viz['Magnitude'], alpha=0.5, s=10)
plt.title('Depth vs Magnitude')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')

# Plot 6: Magnitude distribution by significance
plt.subplot(2, 3, 6)
significant_mags = df_viz[df_viz['is_significant'] == 1]['Magnitude']
non_significant_mags = df_viz[df_viz['is_significant'] == 0]['Magnitude']
plt.hist([non_significant_mags, significant_mags], bins=30, label=['Non-Significant', 'Significant'], 
         alpha=0.7, stacked=True)
plt.title('Magnitude Distribution by Significance')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('/workspace/earthquake_analysis_visuals.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualizations created successfully!")