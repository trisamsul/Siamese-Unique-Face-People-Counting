# Siamese-Unique-Face-People-Counting

This is implementation of Siamese Neural Network for counting people based on their face image. 
Network architecture that i used in this project was implemented from: https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning with a few modification.
This network architecture was provided to extract face features. Then, the extracted features will be used to measure the distance between two compared images.

# Update Log

<b>Beta v1.0</b>

1. New Siamese Architecture
2. New Dataset (Datatrain)
3. New Function for Counting
4. New Function for Recognition

I'm using new datatrain here. This datatrain obtained from: Face94 by Dr Libor Spacek (http://cswww.essex.ac.uk/mv/allfaces/faces94.html). This dataset contains 3080 images in total from 153 different people. This images will be used as datatrain for training process. Each data will have a pair of images and a labels (1.0 for same person, 0.0 for different person).
<b>dataset.py</b> generate permutation 2 in every person (class) for data with same person (label 1.0), and then each images will be paired with 10 different images from 10 different person (label 0.0)
