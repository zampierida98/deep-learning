## Installation of OpenFace via Docker
See [link](https://github.com/TadasBaltrusaitis/OpenFace/wiki#quickstart-usage-of-openface-with-docker-thanks-edgar-aroutiounian-and-mike-mcdermott).

### Usage of OpenFace via command line interface
* `FeatureExtraction` executable is used for sequence analysis that contain a single face.
* `FaceLandmarkVidMulti` is intended for sequence analysis that contain multiple faces.
* `FaceLandmarkImg` executable is for individual image analysis (can either contain one or more faces).

These commands will create a `processed` directory in the working directory that will contain the processed features.

By features we refer to all the features extracted by OpenFace: facial landmarks, head pose, eye gaze, facial action units, similarity aligned faces, and HOG. See [link](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format).

## Logs
### SINGLE-PERSON (`FeatureExtraction`)
* `VID_20220830_161618`: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% Aborted (OpenCV Error)
* `VID_20220830_171739`: 0% 10% 20% 30% 40% Face too small for landmark detection Aborted (OpenCV Error)
* `VID_20220830_171920`: 0% 10% 20% 30% 40% 50% 60% 70% 80% Face too small for landmark detection 90% 100%
* `VID_20220830_172128`: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%
* `VID_20220830_172442`: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

### MULTI-PERSON (`FaceLandmarkVidMulti`)
* `VID_20220830_161618`: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%
* `VID_20220830_171739`: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%
* `VID_20220830_171920`: Killed
* `VID_20220830_172128`: Killed
* `VID_20220830_172442`: Killed

## Outputs
See [link](https://1drv.ms/u/s!ArXDo-v_m_r3uk40MSgk7w8Y-vK9?e=cDkMjD).

## Estrazione dei vettori di gaze
Punto di partenza: [link](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/matlab_runners/Demos/gaze_extraction_demo_vid.m).

Notando che il grafico (ad es. `VID_20220830_172442_gaze_1`) è fitto negli istanti in cui vengono rilevate più persone, è stato diviso il file CSV di output in base alla colonna `face_id` in modo da poter ristampare il grafico con le linee distinte (ad es. `VID_20220830_172442_gaze_2`).

In particolare, i grafici mostrano i valori della direzione dello sguardo dell'occhio. Essi sono espressi in radianti e vengono calcolati come media di entrambi gli occhi per essere convertiti in un formato più facile da usare rispetto ai vettori di gaze (le cui coordinate si trovano nelle colonne `gaze_0_x, gaze_0_y, gaze_0_z` per l'occhio più a sinistra nell'immagine e `gaze_1_x, gaze_1_y, gaze_1_z` per l'occhio più a destra nell'immagine).

Quindi: se una persona guarda da sinistra a destra, ciò comporterà il cambiamento di `gaze_angle_x` (da positivo a negativo); mentre, se una persona guarda in alto, ciò comporterà un cambiamento di `gaze_angle_y` (da negativo a positivo); infine, se una persona sta guardando dritto davanti a sé, allora entrambi gli angoli saranno vicini a 0 (a meno dell'errore di misurazione).
