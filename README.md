# yolo.cpp
an attempt to replicate YOLO architecture using CIFAR-10 dataset from scratch in C++

things to notice:
- tiny_network.h and tiny_network.cpp come from raw written code without using any multithreading using omp.
- since i don't write any CUDA's code for the training so it may takes a lot of time.


i'll appreciate anyone who wants to contribute or giving a feedback to optimize or correcting if it works poorly :)

```
g++ -fopenmp -o main main.cpp cifar10_loader.cpp TinyNn_optimized.cpp `pkg-config --cflags --libs opencv4`
```

