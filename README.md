# Kmeans Cuda
 Cuda C implementation of the Kmeans algorithm


# Specification for sequential implementation of kmeans
#
[Sequential]
How_To_Compile: make
Executable: bin/kmeans
Extra_Args: -p 0

#
# Specification for GPU implementation of kmeans
# using Thrust
#
[Thrust]
How_To_Compile: make
Executable: bin/kmeans
Extra_Args: -p 3

#
# Specification for GPU implementation of kmeans
# using CUDA
#
[CUDA basic]
How_To_Compile: make
Executable: bin/kmeans
Extra_Args: -p 1

#
# Specification for GPU implementation of kmeans
# using Shared Memory
#
[CUDA shared]
How_To_Compile: make
Executable: bin/kmeans
Extra_Args: -p 2

#
# Specification for GPU implementation of kmeans
# Alternatives
#
[Alternatives]
How_To_Compile:
Executable:
Extra_Args:

**Theoretical Program Speedup:**
Calculated using Amdahl's law.
![image](https://github.com/user-attachments/assets/22e77bc1-bc39-47bc-8c1e-a2f523990437)

**Actual Program Speedup E2E:**
