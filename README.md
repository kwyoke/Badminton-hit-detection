# Detecting shuttlecock hit events in professional and amateur badminton videos

In this project, we develop a hit detection algorithm to detect hits in generic monocular videos, agnostic to camera angle and player skill level. We propose to use badminton domain features (court, pose and shuttlecock coordinates) as input to a GRU based model. These coordinates are extracted from each frame, and a sequence of coordinates from 14 consecutive frames is fed into the GRU model to predict three classes: 'no hit', 'hit by near player', or 'hit by far player'. During training, we consider a sequence of 14 frames to be a hit, if a hit is found in the last six frames.

## Contributions
1. We find that our proposed method, even though it is only trained with professional broadcast singles videos, is able to generalize to some extent to amateur videos taken from different camera angles, doubles games, and players of different skill levels. This validates the robustness of badminton domain features (coordinate form) over generic RGB features.
2. We provide annotated datasets of professional singles, amateur singles and amateur doubles matches ![here](https://drive.google.com/drive/folders/13Ja-lZCNNbWelWcb3oP4L4if8khqTfUo?usp=sharing). 
3. We provide manual annotation tools to faciliate annotation of custom datasets under the datasets/ directory.
4. We provide a semi-automatic annotation pipeline in the annotation_pipeline/ directory.

## Demo
Below, we show a few demos of our proposed hit detection algorithm on videos of different camera angles, players of different skill levels, singles and doubles videos.

### Professional video:
[!['pro/test_match1/1_05_02'](https://youtu.be/Sga5BMbK9Qk/maxresdefault.jpg)](https://youtu.be/Sga5BMbK9Qk)

### Amateur singles video:
[!['am_singles/match24/1_05_05'](https://youtu.be/WpQMvr3_JuY/maxresdefault.jpg)](https://youtu.be/WpQMvr3_JuY)

### Amateur doubles video:
[!['am_doubles/match_clementi/doubles5'](https://youtu.be/WpQMvr3_JuY/maxresdefault.jpg)](https://youtu.be/79Vh_RI03KY)

## Datasets
We prepare three sets of annotated matches: Professional singles, amateur singles, amateur doubles. They are available in this ![google drive link](https://drive.google.com/drive/folders/13Ja-lZCNNbWelWcb3oP4L4if8khqTfUo?usp=sharing) and should be placed under the datasets/ directory.

We provide ground-truth annotations of shuttlecock coordinates, hit detections, and player bounding boxes. We also provide manual annotation tools under datasets/.

![Dataset statistics](pics/dataset_stats.png)

## Evaluation
We compare our proposed methods with two baselines, a ResNet image classifier and a rule-based baseline based on comparing the second derivative of the shuttlecock x and y coordinates with an empirically determined threshold. We use mAP as the evaluation metric (see report for more information).

![Table of evaluation comparison](pics/results_table_02.png)

### Performance at various IoU thresholds

#### Different methods, same dataset
![Graph comparison of different methods on the same dataset](pics/differentmethods.png)

#### Different datasets, same method
![Graph comparison of same method on different datasets](pics/perf_different_datasets.png)

#### Detecting near player hits vs far player hits
![Graph comparison of performance on near vs far player hit detection](pics/avg_prec_nearfar.png)

### Qualitative analysis of strengths and weaknesses
Check out these videos for a demonstration of various qualities:
- Mixing up of near and far player hits when camera angle is too wide
  - ![am_singles/match_china2/singles3.mp4 domain](https://youtu.be/ieLmhx0r1PQ)
  - ![am_singles/match_yewtee/singles0.mp4 domain](https://youtu.be/bM7ez-uBKwo)
- Robustness to occluded shuttlecock/ poses
  - shuttlecock occluded, rule-based fails but domain works
    - ![am_doubles/match_yewtee/doubles2.mp4 domain](https://youtu.be/Gle4XFsr6t8)
    - ![am_doubles/match_yewtee/doubles2.mp4 rule-based](https://youtu.be/t_kPVLCtunY)
  - pose occluded, resnet fails but domain works
    - ![am_doubles/match_msia/doubles3.mp4 domain](https://youtu.be/PrToOe11IbI)
    - ![am_doubles/match_msia/doubles3.mp4 resnet](https://youtu.be/-L3xA0hdUU0)
  - Able to tell there is no hit, even when player performs hit action halfway but stops when he realises shuttlecock is going out of court
    ![pro/test_match1/1_05_02.mp4 domain](https://youtu.be/Sga5BMbK9Qk)

## Usage
1. Rally videos and their ground-truth annotations are provided under datasets/ in pro.zip, am_singles.zip, am_doubles.zip, or can be downloaded ![here](https://drive.google.com/drive/folders/13Ja-lZCNNbWelWcb3oP4L4if8khqTfUo?usp=sharing)
2. Manual annotation tools are provided under datasets/ in the scripts label_tool_bbox.py, label_toolV2.py
3. Semi-automatic annotation pipeline for pose and shuttlecock coordinates are in annotation_pipeline/
3. Notebooks for organising datasets into input features and observing dataset statistics can be found in annotation_pipeline/organise_input_features.ipynb and annotation_pipeline/dataset_stats.ipynb
4. Notebooks for training the proposed domain based algorithm and ResNet are found in domain_rnn.ipynb and ResNet_baseline.ipynb respectively. They take in input features from the directory input_features/
5. Notebooks for processing classification probabilities from the proposed algorithms can be found in hit_detection/process_pred_probs.ipynb
6. The notebook for rule-based baseline method can be found in hit_detection/rule_baseline.ipynb
7. Pretrained weights for the proposed domain method can be found in mm_weights/
8. Pretrained weights for ResNet can be found in resnet_data

A small GPU is required for running the semi-automatic annotation pipeline, as well as for training the proposed GRU network and ResNet. The computational load is fairly light, see details in the training notebooks.

## References
The following references were immensely useful for this project.
1. ![MonoTrack: Shuttle trajectory reconstruction from monocular badminton video](https://arxiv.org/pdf/2204.01899)
on using badminton domain features for hit event detection
2. ![TrackNetV2: Efficient Shuttlecock Tracking Network](https://ieeexplore.ieee.org/document/9302757) on tracking shuttlecock with deep learning, as well as providing the TrackNetv2 dataset which formed the basis of our Professional dataset.

## Full report
The full details are documented in the pdf report.

## Future directions
1. Domain Adaptation to improve generalisation.
2. Multimodal feature learning, possibly combine audio and rgb features with domain coordinates.
3. Larger and more varied training dataset.
4. Extension to other aspects of badminton video analysis, including stroke classification, strategy analysis etc.