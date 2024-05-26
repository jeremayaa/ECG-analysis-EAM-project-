import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy import signal
import seaborn as sns


persons = ['JP', 'WR', 'AS', 'JS']

def filter_ekg(ekg, low_band=2, high_band=30, order=10):
    sos = signal.butter(order, high_band, 'lp', fs=1000, output='sos')
    ekg = signal.sosfilt(sos, ekg)

    # High pass filter
    sos = signal.butter(order, low_band, 'hp', fs=1000, output='sos')
    ekg = signal.sosfilt(sos, ekg)
    return ekg    

heart_rates = []

for j, name in enumerate(persons):
    # Read data
    chill = pd.read_csv(f'Dane/{name}/ekg_spoczynek.lvm', sep='\t', decimal=',').iloc[:, 1].values
    pushups = pd.read_csv(f'Dane/{name}/ekg_pompki.lvm', sep='\t', decimal=',').iloc[:, 1].values
    squats = pd.read_csv(f'Dane/{name}/ekg_przysiady.lvm', sep='\t', decimal=',').iloc[:, 1].values
    jumps = pd.read_csv(f'Dane/{name}/ekg_podskoki.lvm', sep='\t', decimal=',').iloc[:, 1].values

    data = {'chill': chill, 'pushups': pushups, 'squats': squats, 'jumps': jumps}

    fig, axs = plt.subplots(4, 1, figsize=(12, 12))

    plot_window = (0, 15000)

    person_heart_rate = {'person': f'person {j}'}

    # Iterate over the key-value pairs in the data dictionary
    for i, (key, value) in enumerate(data.items()):
        
        ekg = value[plot_window[0]:plot_window[1]]

        # Comment/Uncomment if necessary
        ekg = filter_ekg(ekg)
        
        signals, info = nk.ecg_process(ekg, sampling_rate=1000, method='pantompkins1985')

        mask = (info['ECG_R_Peaks'] > plot_window[0]) & (info['ECG_R_Peaks'] < plot_window[1])
        QRS = info['ECG_R_Peaks'][mask]

        fig.suptitle(f"Person {j+1}")
        
        # Comment/Uncomment if necessary
        # axs[i].scatter(QRS, (max(ekg)+500)*np.ones(len(QRS)), color='r')
        axs[i].plot(ekg)
        axs[i].set_title(key)

        # Calculate average heart rate
        rr_intervals = np.diff(QRS) / 1000  # Convert to seconds
        avg_heart_rate = 60 / np.mean(rr_intervals)  # Convert to beats per minute
        person_heart_rate[key] = avg_heart_rate

    heart_rates.append(person_heart_rate)

    # plt.tight_layout()  
    # Comment/Uncomment if necessary
    # plt.savefig(f'Plots/Raw data without QRS detections, person {j+1}.png')

# Convert heart rate data to DataFrame and display
heart_rate_df = pd.DataFrame(heart_rates)
print(heart_rate_df)



# Create the heatmap
heart_rate_df.set_index('person', inplace=True)
plt.figure(figsize=(10, 6))
sns.heatmap(heart_rate_df, annot=True,fmt='.2f', cmap='coolwarm', linewidths=.5)

# Add a title for better understanding
plt.title('Częstotliwość bicia serca AHR [bpm]')

# Show the plot
plt.show()