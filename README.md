
## **Data Summary:**

Dataset consists of 3-channel surface electromyography (sEMG) recorded from under the chin (submental muscles) and chest (intercostal and diaphragm muscles) from 11 study participants (8 control, 3 throat cancer).
Each participant attended 4 sEMG recording sessions, each session consisting of the following:

    15 swallows (5 dry, 5 water, 5 banana).
    3 cough recordings, each recording comprising 5 coughs.
    3 speech recordings, each comprising 10 spoken sentences from the Harvard sentences (IEEE, "IEEE Recommended Practice for Speech Quality Measurements," IEEE Trans. Audio Electroacoust., vol. 17, no. 3, pp. 225-246, 1969).
    6 movement recordings comprising the actions of standing, walking, reaching, twisting and sitting.
    1 baseline recording, participant remained still for 60 seconds while baseline signals were recorded.

Sound was recorded concurrently during swallow and speech recordings using a contact microphone placed over the cricoid cartilage. Pneumotachometry (airflow) was recorded during coughing.
P1-8: Control participants P9-11: Thoart cancer participants

---
## **Data Folders and File Notes:**

Dataset contains both raw and processed data (signals have been normalized to baseline recording and excessive artefacts removed).

1 folder exists per session: "P1_S1" corresponds to Participant 1 Session 1 data. 1 .csv file exists per recording. From left to right .csv columns correspond to:

    1. Submental sEMG, 
    2. Intercostal sEMG, 
    3. Diaphragm sEMG, 
    4. Pneumotachometry, 
    5. Contact Microphone, 
    6. Class Label.
        mV,mV,mV,cmH20,V,N/A (Units for raw data)

Class labels are as follows:
- 0 - Null (anything outside the other classes)
- 1 - Swallow phase 1 (preparation activity for swallowing such as chewing, sipping etc.)
- 2 - Swallow phase 2 (swallow reflex, larynx elevation following submental muscle contraction)
- 3 - Cough
- 4 - Speech


Movement recordings start with standing and end with sitting. The order of walking, twisting and reaching is randomised and summarised by the recording title e.g. "08_reach_twist_walk.csv" indicates an order of reach-twist-walk.

Files ending in "SW" indicate an out-of-protocol swallow took place e.g. participant swallowed during a cough recording.

Swallow files ending in "N2" or "N3" contain 2 or 3 swallows respectively.

>1 baseline recording may exist in a session. If electrodes became detached during a session the baseline recording was repeated and subsequent recordings in the session normalised according to the latter baseline.
---
## **Hardware Setup:**

    - Two submental electrodes (EL513, 10 mm diameter, BIOPAC Systems UK) were  placed on the midline, posterior to the mental protuberance, with 20 mm  interelectrode distance. 
    - Three electrodes (EL503, 11 mm diameter,  BIOPAC) were placed on the right 9th/10th intercostal space close to the  anterior axillary line, with 35 mm interelectrode distance. The  posterior two electrodes formed the intercostal recording dipole. The anterior electrode and a single electrode placed on the left 9th/10th intercostal space formed the diaphragm recording dipole.
    - Two reference  electrodes (EL503) were placed on the midline over the sternum. 
    - Two  wireless EMG recorders (BIOPAC BN-EMG2 BioNomadix, 2,000 Hz sampling  rate, 2,000× gain, 5 to 500 Hz bandpass filter) were placed at the waist  and on the head to minimise relative cable length and motion artefacts.


---
Reference Material:

_J. McNulty, M. Birchalland A. Vanhoestenberghe, “sEMG of Swallowing, Coughing and Speech”. University College London, 25-Oct-2023, doi: 10.5522/04/24297766.v1._

Dataset: [sEMG_of_Swallowing_Coughing_and_Speech](https://rdr.ucl.ac.uk/articles/dataset/sEMG_of_Swallowing_Coughing_and_Speech/24297766?file=42865813)
