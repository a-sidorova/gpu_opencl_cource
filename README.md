# Programming on new Architecture-1 (GPU). Autumn. 2021
This repository stores laboratory works of Programming on new Architecture-1 (GPU) course of autumn 2021.

*Language*: C/C++

*Tools*: OpenCL

### Themes of labs:
1. lab01 - *Print thread info and addition of src data and global ID of thread.*


### Build the project with `CMake`
Main folder (root) of repository.

1) Configure the build:

  ```
  mkdir build && cd build
  cmake -DLAB#=ON ..
  ```
*Help on CMake keys:*
- `-DLAB#=ON` enable `Lab#` project (`#` - number of laboratory work).

*A corresponding flag can be omitted if it's not needed.*

2) Build the project:
  ```
  cmake --build .
  ```


© Copyright Sidorova Alexandra, 2021
