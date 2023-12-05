# Genre Reveal HPC Project

This project contains multiple implementations of the kmeans algorithm for clustering songs by various categories

### CHPC Dependencies
```
module load gcc/8.5.0 && module load intel-mpi  
module load cuda/12.2.0  
```

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

### Running the project implementations:
If the user does not provide arguments, the default runtime configuration of the executables is as follows:
```
    category 1  : danceablility
    category 2  : loudness
    category 3  : instrumentalness
    writeToFile : true
    k           : 1
    epochs      : 10
    numThreads  : 1
```
Otherwise, each executable can be run with the following arguments:
```
./ExecutableName <category 1> <category 2> <category 3> <writeToFile (t/f)> <k> <epochs> [numThreads]
```
Within your build directory, 
for Serial run:
```
./dist/bin/Serial
```
#### Parallel
for Shared Memory CPU run:
```
./dist/bin/openmp_exec
```
for Shared Memory GPU run:
```
./dist/bin/SharedMemGpu
```
for Distributed Memory CPU run:
```
mpiexec -n <# of cores> ./dist/bin/DistMemCpu
```
for Distributed Memory GPU run:
```
mpiexec -n <# of cores> ./dist/bin/DistMemGpu
```


### Visualization
After running one of the above data is stored in ```tracks_output.csv```

Within the build folder run
```
python plot.py
```
for a visualization of the output data/classification.
