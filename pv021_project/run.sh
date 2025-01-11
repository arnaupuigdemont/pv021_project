echo "Adding some modules"

module add gcc-10.2

echo "#################"
echo "    COMPILING    "
echo "#################"

# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2 -o network
g++ -Wall -Werror -std=c++17 src/main.cpp src/matrix.cpp src/dataset.cpp src/network.cpp src/layer.cpp -g -o network -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -fopenmp

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 ./network
