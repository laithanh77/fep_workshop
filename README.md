# fep_workshop

Simple FEP implementation using CHARMM/BLaDE and parallelized with slurm job arrays

Instructions:

Submit fep job through `sbatch run_fep.sb folder_name/`

Windows are created based on job_array, and lambda interval is computed by evenly dividing 1 by number of task array jobs

Each worker (job array) will create files with the name folder_name_wX where X is the job array ID

The new directory will contain:
out/: a folder containing the redirected CHARMM output (.out)
output/: a folder containing the final structures after dynamics (.crd), and the perturbation calculations (.npy)
dcd/: a folder containing dcd files (.dcd)
res/: a folder containing restart files (.res)

After running FEP (should take 5-8 minutes for 5 ns), run the perturbation using `sbatch compute_forward_perturbations.sb folder_name/`

This script works as a task array. Each worker will load in the corresponding dcd file, compute the reference ensemble energies for each frame and the perturbation energy for the next window.
Then it saves an array that contains the delta E between the perturbation energy and the reference energy.
