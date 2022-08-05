# Deep Learning Model to Detect Emotions through Brain-Waves
This project is part of our Biomedical Engineering course project work. Emotional response was recorded via the Ultracortex Mark IV EEG helmet on the open BCI software, divided in three categories of negative, neutral and positive and trained on a deep learning model. 

# What are Brain Waves? 
Brainwaves are electrical impulses in the brain. An individual’s behaviour, emotions, and thoughts are communicated between neurons within our brains. All brainwaves are produced by synchronised electrical pulses from masses of neurons communicating with each other. Our brainwaves occur at various frequencies. Some are fast and some are slow. They are measured in cycles per second or hertz (Hz).
The classic names of these EEG bands are: 
* Delta Brainwaves (1-3 Hz)
* Theta Brainwaves (4-7 Hz)
* Alpha Brainwaves (8-12 Hz)
* Beta Brainwaves (13 – 38 Hz) 
* Gamma Brainwaves (39 – 42 Hz)

Each of us, however, always has some degree of each of these brainwave bands present in different parts of our brain. Delta brainwaves will also occur when areas of the brain go “off line” to take up nourishment. If we are becoming drowsy, there are more delta and slow theta
brainwaves creeping in. If we are inattentive to external things and our mind is wandering, there is more theta present. If we are exceptionally anxious and tense, an excessively high frequency of beta brainwaves is often present.

# How does EEG detect brain activites?
An electroencephalogram (EEG) is a test that measures electrical activity in the brain using small, metal discs (electrodes) attached to the scalp. Brain cells communicate via electrical impulses and are active all the time, even during asleep. This activity shows up as wavy lines on an EEG recording. 

# EEG Helmet
An EEG headset is a wearable device for electroencephalography, a monitoring method to record the electrical activity of the brain. EEG sensors in headsets place electrodes along the scalp to detect brain activity. Analyzing EEG data supports the study of cognitive processes. Doctors can use EEG to diagnose medical issues, researchers can use this method to understand brain processes, and individuals can use EEG to improve their productivity and wellness via monitoring their moods and emotions, developers can use EEG for BCI to execute direct mental commands in app development and many other use cases.
There are a few different types of EEG headsets. Comparison is often drawn between dry electrode EEG headsets and wet electrode arrays. Wet electrodes use a conductive gel, saline fluid or other material to improve signal quality. This ensures the EEG device captures high-quality data. Most dry EEG headsets provide good quality readings, so the conductive gel is not normally required unless high- quality data is required for specific clinical reasons or high accuracy.

# Open BCI 
OpenBCI is an open-source brain-computer interface platform.
OpenBCI boards can be used to measure and record electrical activity produced by the brain (EEG), muscles (EMG), and heart (EKG), and is compatible with standard EEG electrodes. The OpenBCI boards can be used with the open source OpenBCI GUI, or they can be integrated with other open-source EEG signal processing tools.

![emotion1](https://user-images.githubusercontent.com/102278418/183085800-9f31d8e1-5d58-4975-9d4c-bba85416980c.jpg)


# Ultracortex Mark IV Helmet
The Ultracortex is an open-source, 3D-printable headset intended to work with any OpenBCI Board. It is capable of recording research-grade brain activity (EEG), muscle activity (EMG), and heart activity (ECG). It is not designed for transcranial stimulation. This headset is designed to receive EEG signals only. The Ultracortex Mark IV is capable of sampling up to 16 channels of EEG from up to 35 different 10-20 locations.

![emotion6](https://user-images.githubusercontent.com/102278418/183085821-70105a86-0fe2-46f2-a450-a229e3c2f0dc.jpg)


## Dependencies
* Python (3.6 or higher)
* Pandas
* Keras 
* Tensorflow 
* Numpy
* Matplotlib

### This project is ran/tested on Google Colab. 

# Dataset 
* For this dataset, brainwaves of a user during certain movie scenes and used them to calculate whether the same brain wave frequency would be emitted by different users over the same scene. Emotional response was categorised between positive, negative and neutral.
* Dataset was collected via Open BCI software using an EEG headset with 8 sensors.
* With each scensor, the particular brainwave emitted was calculated by the sensor nodes on the EEG headset and recorded on the Open BCI software. The waveform data is then pre-processed into arrays and used for the deep learning model.
### FFT Representation of dataset
![emotion2](https://user-images.githubusercontent.com/102278418/183085924-8938c8cd-d5a7-4dc8-9bab-25c2d2be5ed2.jpg)


# Model Architecture

## CNN Model 
* A CNN (Convolutional Neural Networks) based model was used as our base algorithm. 
* Softmax activation function was to determine the accuracy.
* Epochs Trained on : 50

# Results
Accuracy: 95%  

### Train vs Test Accuracy Graph
![emotion5](https://user-images.githubusercontent.com/102278418/183086020-1c528964-9bf0-4ba9-bf48-4646a207f7c9.jpg)


### Train vs Test Loss Graph
![emotion4](https://user-images.githubusercontent.com/102278418/183086031-2025cd6a-32ba-46db-85f6-421b34180e74.jpg)


### Confusion Matrix
![emotion3](https://user-images.githubusercontent.com/102278418/183085989-9ce33ba4-8097-438d-abff-f1b94686b16b.jpg)



# Team Members
* Rohan Srinivasan (Me) (Linkedin: https://www.linkedin.com/in/rohan-srinivasan-2457591b1/)
* Peddi Giridhar (https://github.com/Giridhar4) (Linkedin: https://www.linkedin.com/in/giridhar-peddi-68485519b/)
* Simone Singh (Linkedin: https://www.linkedin.com/in/simone-singh-29946a143/)
* Sanjana Golaya (Linkedin: https://www.linkedin.com/in/sanjana-golaya/)
