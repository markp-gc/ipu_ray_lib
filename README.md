# Ray-Tracing on Graphcore IPUs

![Example output image](images/example.png "Images path traced on IPU.")

This is version 2 of an experimental ray-tracer/path-tracer for IPU (old version is [here](https://github.com/markp-gc/ipu_path_trace)). This new version is completely re-architected.

Key improvements over the original are:

- Triangle meshes are now supported.
- An in SRAM acceleration structure is now supported:
  - A BVH built by Embree is compacted to fit in IPU SRAM.
- Rays are streamed from external DRAM which has the following benefits:
  - No limit on the resolution of the images rendered as the entire result does not need to fit in SRAM at once.
  - Improved memory efficiency as only the minimal number of rays are kept on chip at once.
- Streaming rays on/off chip happens in parallel with ray/path-tracing using overlapped I/O.
- The implementation also contains two reference CPU code paths for comparison:
  - One uses Embree.
  - The other uses identical code to the IPU kernels but runs multi-threaded on CPU.
  - The CPU implementation also aids debugging of IPU code as it can be run single threaded on the host and the program can compare IPU's ray-tracing AOVs to Embree's.
- Path contributions are not stored and deferred: throughput is calculated during path-tracing which frees up more on-chip memory.
- Hardware random numbers are generated inline in the path tracing kernel using the IPU's built-in RNG. This simplifies the code and also reduces SRAM consumption.
- Avoid using sin/cos from std library (see [ext/math](ext/math/README.md)):
  - Increases path-trace rate by 1.5x.
  - Saves ~9Kb of double emulation code per tile.
- Multi-IPU rendering is now implemented using replicated graphs. The replicas process ray-data-parallel streams from DRAM.
  - This greatly speeds up graph construction and compile time: a multi-IPU path tracing graph can now be compiled from scratch in ~10 seconds.

Note: some fancy features of the old version have not been ported to the new version:
- Neural rendering (neural HDRI lighting) not yet implemented.
- Remote-user interface with live preview not yet supported.

## Try it immediately in a free notebook

This code has been tested on IPU PODs running Ubuntu 20 and Poplar SDK version 3.1. It depends on a number of apt packages therefore the
simplest way to try it out is launching a pre-built docker container on a cloud service. Paperspace, for example, provides free IPU-POD4
machines. Click this link and follow the instructions in the README.ipynb notebook:

[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/markp-gc/gradient_poplar_raytracer?container=mpupilli/poplar_paperspace&machine=Free-IPU-POD4&file=README.ipynb)

## Setting up your own machine:

If you would prefer an alternative to [Paperspace](https://www.paperspace.com/graphcore) You can use this [Dockerfile](https://github.com/markp-gc/docker-files/blob/main/graphcore/poplar_dev/Dockerfile) on a cloud service of your choice or even manually install the apt dependencies listed in it on your own system. In this case you will need to follow instructions for your cloud/system to set it up yourself.

Once your system is configured you can now clone and build this repository. The build uses CMake:
```
git clone --recursive https://github.com/markp-gc/ipu_ray_lib
mkdir -p ipu_ray_lib/build
cd ipu_ray_lib/build
cmake -G Ninja ..
ninja -j64
```

### Run the application

The application loads mesh data using the [Open Asset Import Library](https://github.com/assimp/assimp).
Currently meshes need to fit on tile, the provided mesh is small enough:
```
./test -w 1440 -h 1440 --mesh-file ../assets/monkey_bust.glb --render-mode path-trace --visualise rgb --samples 1000 --ipus 4 --ipu-only
```

After about 30 seconds this command will output an EXR image 'out_rgb_ipu.exr' in the build folder.
If you want to quickly tonemap the HDR output and convert to PNG run this command:
```
pfsin out_rgb_ipu.exr | pfstmo_mai11 | pfsout /notebooks/box.png
```

If you want to render a CPU reference image remove the option `--ipu-only` but be aware it will
take much much longer to render.

If you just want to compare AOVs between CPU/Embree/IPU you can
change to a quicker render mode. E.g. to compare normals:
```
./test -w 1440 -h 1440 --mesh-file ../assets/monkey_bust.glb --render-mode shadow-trace --visualise normal --ipus 4
```
If you compare 'out_normal_cpu.exr', 'out_normal_embree.exr', and 'out_normal_ipu.exr' you should find they match closely.

For a list of all command options see `./test --help`.
