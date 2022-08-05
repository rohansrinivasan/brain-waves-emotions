# Loads the dataset used from drive
# Dataset consists of three modalities, i.e. Surface Electro-myogram or sEMG, tri-axis gyroscope and tri-axis accelerometer.
# Signals were captured using six Delsys wireless sensors, consisting of one sEMG sensor and one IMU containing a tri-axis accelerometer and a tri-axis gyroscope each. 

data=pd.read_csv('/content/emotions.csv')
data

# features plot of fft data
sample = data.loc[0, 'fft_0_b':'fft_749_b']
plt.figure(figsize= (16,10))
plt.plot(range(len(sample)),sample)
plt.title('Features fft_0_b through fft_749_b')
plt.show()