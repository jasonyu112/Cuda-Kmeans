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
![image](https://github.com/user-attachments/assets/d06ef97c-b054-4a2e-b3c1-f12e15a62b63)

**Per Iteration Speedup:**                
![image](https://github.com/user-attachments/assets/323b3cf5-5b3f-40fb-9583-cb1251883cf5)

**Data Transfer Times E2E:**                    
![image](https://github.com/user-attachments/assets/91d731f2-8121-44d1-8ae3-c14d70297b80)


