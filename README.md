# Genre Reveal HPC Project

This project contains multiple implementations of the kmeans algorithm for clustering songs by various categories

### Cloning and Building the project:
```
git clone https://github.com/kodamums/GenreReveal
cd GenreReveal
mkdir build
cd build
cmake ../../GenreReveal
make
```
download tracks_features.csv from https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs and add it to the build folder

### Running the project:
Within your build directory, run:
```
./GenreReveal
python plot.py
```

