[Original Website](https://rdr.ucl.ac.uk/articles/dataset/sEMG_of_Swallowing_Coughing_and_Speech/24297766?file=42865813)

Notes on dataset:

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

P1-8: Control participants
P9-11: Thoart cancer participants

4 recording sessions for each participant

During each session the following were recorded:

    15 recordings of swallowing: 5 dry; 5 water; 5 banana
    3 recordings of coughing, with each recording containing 5 coughs.
    3 recordings of speech, in which the participant read aloud 10 sentences from the Harvard sentences (IEEE, "IEEE Recommended Practice for Speech Quality Measurements, " IEEE Trans. Audio Electroacoust., vol. 17, no. 3, pp. 225-246, 1969).
    6 recordings of movements typical of daily life (standing, sitting, reaching, twisting and walking).
    1 baseline recording, participant remained still for 60 seconds while baseline signals were recorded.

Additionally, sound was recorded during swallow and speech recordings using a contact microphone placed over the cricoid cartilage. Pneumotachometry (airflow) was recorded during coughing.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Data is available in both raw and processed (sEMG normalized to baseline recordings, excessive artefacts removed) formats. Each .csv is structured as follows:


EMG-submental,EMG-intercostal,,EMG-diaphragm,pneumotachometry, contact, microphone, class labels
mV,mV,mV,cmH20,V,N/A (Units for raw data)

Class labels are as follows:
0 - Null (anything outside the other classes)
1 - Swallow phase 1 (preparation activity for swallowing such as chewing, sipping etc.)
2 - Swallow phase 2 (swallow reflex, larynx elevation following submental muscle contraction)
3 - Cough
4 - Speech
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sampling rate for EMG, pneumotachometry & sound = 2000 Hz

Pneumotachometry conversion from cmH20 to L/min: L/min = cmH20 x 479

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

In pneumotachometry data: Positive values = exhalation, negative values = inhalation.

Movement recordings start with standing and end with sitting. The order of walking, twisting and reaching is randomised and summarised by the recording title e.g. "08_reach_twist_walk.csv" indicates an order of reach-twist-walk.

Files ending in "SW" indicate an out-of-protocol swallow took place e.g. participant swallowed during a cough recording.

Swallow files ending in "N2" or "N3" contain 2 or 3 swallows respectively.

>1 baseline recording may exist in a session. If electrodes became detached during a session the baseline recording was repeated and subsequent recordings in the session normalised according to the latter baseline.
