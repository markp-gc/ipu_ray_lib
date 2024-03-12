# Ray tracing on Graphcore IPUs

![Example output image](images/example.png "Images path traced on IPU.")

This project is a redesign of an experimental ray/path tracer for Graphcore Intelligence Processing Units (IPUs). Key features are:
* Executes on Graphcore IPUs (not nominally designed for rendering).
* It combines path tracing with "neural rendering": a high-dynamic-range (HDR) environment light is encoded in a small neural image field (NIF).
* The neural network weights, scene description and bounding volume hierachy (BVH) reside entirely in on-chip SRAM.

The old version is [here](https://github.com/markp-gc/ipu_path_trace). This new version is completely re-architected with the aim of making it more flexible and easier to extend. There has also been some attempt at making it interoperable with Embree (which is used to build an initial BVH).

## New software architecture

Key improvements of this implementation over the original are:

- Triangle meshes are now supported.
- An in SRAM acceleration structure is now supported:
  - A BVH built by Embree is compacted to fit in IPU SRAM.
  - Compact BVH nodes are partially stored at half precision with no loss in ray-tracing precision.
- Rays are streamed from external DRAM which has the following benefits:
  - No limit on the resolution of the images rendered as the entire result does not need to fit in SRAM at once.
  - Improved memory efficiency as only the minimal number of rays are kept on chip at once.
- Streaming rays on/off chip happens in parallel with ray/path-tracing using overlapped I/O.
- Path contributions are not stored and deferred: throughput is calculated during path-tracing which frees up more on-chip memory.
- Hardware random numbers are generated inline in the path tracing kernel using the IPU's built-in RNG. This simplifies the code and also reduces SRAM consumption.
- Avoid using sin/cos from std library (see [ext/math](ext/math/README.md)):
  - Increases path-trace rate by 1.5x.
  - Saves ~9Kb of double emulation code per tile.
- Multi-IPU rendering is now implemented using replicated graphs. The replicas process ray-data-parallel streams from DRAM.
  - This greatly speeds up graph construction and compile time: a multi-IPU path tracing graph can now be compiled from scratch in ~10 seconds.
- The implementation also contains two reference CPU code paths for comparison:
  - One uses Embree.
  - The other uses identical code to the IPU kernels but runs multi-threaded on CPU.
  - The CPU implementations aid debugging of IPU code. E.g. it is easier to step through a single threaded program on CPU and we can compare IPU's ray-tracing AOVs to Embree/CPU references.

## Setting up your IPU machine:

If you have access to your own IPU machine you can use this [Dockerfile](https://github.com/markp-gc/docker-files/blob/main/graphcore/poplar_dev/Dockerfile) on a cloud service of your choice or even manually install the apt dependencies listed there in a bare metal system. In either case you will need to follow instructions for your cloud/system to set it up yourself (e.g. install the Poplar SDK and configure VIPU to see at least 4 IPUs.)

Once your system is configured you can clone and build this repository. The build uses CMake:
```
git clone --recursive https://github.com/markp-gc/ipu_ray_lib
mkdir -p ipu_ray_lib/build
cd ipu_ray_lib/build
cmake -G Ninja ..
ninja -j64
```

### Run the application

Ray data is distributed across all tiles (cores) but the scene data (BVH) is currently replicated across all tiles. This means meshes need to fit on one tile for now. You can specify your own scenes using the `--mesh-file` option. There is a built-in scene which is rendered if no file is specified:

```
./trace -w 1440 -h 1440 --render-mode path-trace --visualise rgb --samples 1000 --ipus 4 --ipu-only
```

After about 30 seconds this command will output an EXR image 'out_rgb_ipu.exr' in the build folder.
If you want to quickly tonemap the HDR output and convert to PNG run this command:
```
pfsin out_rgb_ipu.exr | pfstmo_mai11 | pfsout tonemapped.png
```

If you want to render a CPU reference image remove the option `--ipu-only` but be aware it will
take much much longer to render.

If you just want to compare AOVs between CPU/Embree/IPU you can
change to a quicker render mode. E.g. to compare normals:
```
./trace -w 1440 -h 1440 --render-mode shadow-trace --visualise normal --ipus 4
```
If you compare 'out_normal_cpu.exr', 'out_normal_embree.exr', and 'out_normal_ipu.exr' you should find they match closely.

For a list of all command options see `./trace --help`.

#### Rendering other scenes

With the `--mesh-file` option the program will attempt to load any file format
supported by the [Open Asset Import Library](https://github.com/assimp/assimp).
It will attempt to interpret imported material properties into one of the
supported materials (currently a bit limited). Until this is improved an example
DAE file that has been exported from [Blender](https://www.blender.org) is included:
this is a human readable file so that you can try to work out how to export your own
scenes by inspecting it. You can render the scene from the DAE file like this
(turning on logging to see how materials are being interpreted):

```
./trace -w 1440 -h 1440 --render-mode path-trace --visualise rgb --samples 4000 --ipus 4 --ipu-only --mesh-file ../assets/test_scene.dae --load-normals --log-level debug
```

## Train your own environment lighting network

The neural environment light uses a neural image field (NIF) network. These are MLP based image approximators and are trained using Graphcore's NIF implementation: [NIF Training Scripts](https://github.com/graphcore/examples/tree/master/vision/neural_image_fields/tensorflow2).
Before you start a training run you will need to source an equirectangular-projection HDRI image (e.g. those found here are suitable: [HDRIs](https://polyhaven.com/hdris)). Download a 2k or 4k image and pass it to the NIF traning script('--input'). You can play with the hyper parameters but the parameters below are a balanced compromise between size, computational cost and quality:

```
git clone https://github.com/graphcore/examples.git
cd examples/vision/neural_image_fields/tensorflow2
pip install -r requirements.txt
python3 train_nif.py --train-samples 8000000 --epochs 1000 --callback-period 100 --fp16 --loss-scale 16384 --color-space yuv --layer-count 6 --layer-size 320 --batch-size 1024 --callback-period 100 --embedding-dimension 12 --input input_hdri.exr --model nif_models/output_nif_fp16_yuv
```

The trained keras model contains a subfolder called `assets.extra`, give that path to the path tracer using the `--nif-hdri` command line option.

## Testing and development

If you want to modify the library we recommend you run through the testing notebook as this gives
a more detailed explanation of the application ![LITERATE_TEST.ipynb](LITERATE_TEST.ipynb).

## Double precision

First and second generation IPUs do not have hardware support for double precision, however C++ code using double's will still
compile and run using LLVM's software emulation library. Double precision is typically used in the ray-triangle intersection
test when the intersection point lies on the triangle boundary (within single precision). In that case the computation falls
back to double to get a "water tight" intersection test at the edges. If this level of precision is important to you there is a
CMake configuration option to enable fall back to doubles: to enable it add this to the command line at configuration time
`-DALLOW_DOUBLE_FALLBACK=1` (by default it is disabled).

Note: there is a small, scene dependent, performance penalty and the double emulation code consumes extra tile memory (~3.7 KiB per tile).
Depending on the scene, the performance penalty has been observed to be between a 1% and 6% lower path trace rate (the double code path is
executed relatively infrequently).
