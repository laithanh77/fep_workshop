import os
import shutil
import sys
import numpy as np
import mdtraj as md
import pandas as pd
import time

import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
from pycharmm.charmm_file import CharmmFile
from pycharmm.lingo import charmm_script
from pycharmm.lib import charmm as libcharmm
from pycharmm.select_atoms import SelectAtoms
from pycharmm.lingo import get_energy_value
import argparse


parser = argparse.ArgumentParser(description="blah blah don't use cmd line args if you are using mpirun. these args are meant for job arrays!")
parser.add_argument("--rank", type=int, default=0,
                    help="worker rank. do not use mpirun if this is used. one worker must be rank 0")
parser.add_argument("--size", type=int, default=1,
                    help="number of workers. do not use mpirun if this is used.")
parser.add_argument("--path", type=str)
args = parser.parse_args()

if args.rank is not None:
    rank = args.rank

if args.size is not None:
    size = args.size

def is_factor(n):
    if (n % 2 != 0): return False  # favors even number
    while n:
        flag = False
        for x in (2,3,5):
            if n % x == 0:
               n = n / x
               flag = True
               break

        if flag: continue
        break

    if n == 1: return True
    return False

def checkfft(n, margin = 5):
    n = int(n) + margin
    while 1:
        if is_factor(n): break
        else: n += 1
    return n

def main():
    #########################
    # SYSTEM SPECIFICATIONS #
    #########################
    temperature = 298.15 
    dt = 0.004
    dt_equil = 0.004
    isvfrq = None  # restart file saving frq
    nprint = 1000 
    nsavc = int(10/dt)
    nequil = int(500/dt)  # equilibration steps
    total_steps = int(5000/dt)  # production steps
    n_frames = int(total_steps/nsavc)

    # across the workers, evenly split up lambda grid
    lambda_values = np.linspace(0, 1, args.size)
    n_ligs = 2
    
    path = args.path
    lig_name = args.path.split(".")[0].replace("/", "")
    dcd_filename = f"{lig_name}_w{rank}"
    filename = f"perturbation_w{rank}"
    psf_start = f"pdb/hybrid_ligands_solvated_hmr.psf"
    coor_start = f"pdb/hybrid_ligands_solvated.crd"

    ###################
    # PREPARE WORKDIR #
    ###################
    # copy this file and slurm script (if possible) to workdir for record-keeping
    if rank == 0:
        this_file = os.path.basename(__file__)
        shutil.copy(this_file, f'{path}/.')
        try:
            shutil.copy(this_file.split(".")[0]+".sb", f'{path}/.')
        except:
            pass

    out_file = pycharmm.CharmmFile(file_name=f"{path}/out/{filename}.out", file_unit=35, formatted=True, read_only=False)
    charmm_script(f"outu 35")
 
    ##########################
    # LOAD TOPPAR AND SYSTEM #
    ##########################
    charmm_script("bomblev -2")
    read.rtf("pdb/top_all36_cgenff.rtf")
    read.prm("pdb/par_all36_cgenff.prm", flex=True)
    charmm_script("stream pdb/toppar_water_ions.str")
    charmm_script(f"stream pdb/hybrid_ligands.str")

    read.psf_card(psf_start) 
    read.coor_card(coor_start) 


    ##############################################
    # PIN COMMON CORE ATOMS TOGETHER WITH NOES   #
    # SINCE WE ARE DOING LIGAND OVERLAY TOPOLOGY #
    ##############################################
    common_core_atom_mapping = [("C3", "C4"), ("C4", "C5"), ("C5", "C1"), 
                                ("C6", "C2"), ("C1", "C2"), ("C2", "C3")]

    for pin in common_core_atom_mapping:
        charmm_script(f'''
            NOE
            assign -
            sele segid LIGA .and. resname MOLB .and. type {pin[0]} end -
            sele segid LIGA .and. resname MOLP .and. type {pin[1]} end -
            kmin 200 rmin 0.01 kmax 200 rmax 0.01 fmax 9999
            print analysis
            END''')




    #######################
    # BUILD PBC AND IMAGE #
    #######################
    coor.orient(norotation=True)
    stats = coor.stat()

    xsize = stats['xmax'] - stats['xmin']
    ysize = stats['ymax'] - stats['ymin']
    zsize = stats['zmax'] - stats['zmin']
    box_size = max(xsize, ysize, zsize)
    pmegrid = checkfft(n=np.ceil(box_size/2.0)*2, margin=0)

    origin = 0

    xyz = coor.get_positions()
    xyz["x"] += origin 
    xyz["y"] += origin 
    xyz["z"] += origin
    coor.set_positions(xyz)

    crystal.define_cubic(box_size)
    crystal.build(box_size/2.0)

    image.setup_residue(origin, origin, origin, "TIP3")
    image.setup_segment(origin, origin, origin, "LIGA")

    ##############################
    # SET UP NBONDS AND MINIMIZE #
    ##############################
    cutnb = min(box_size/2, 12)
    cutim = cutnb
    ctofnb = cutnb - 1.0
    ctonnb = cutnb - 3.0

    nbonds_dict = {'cutnb':cutnb,'cutim':cutim,
               'ctonnb':ctonnb,'ctofnb':ctofnb,
               'atom':True,'vatom':True,
               'cdie':True,'eps':1.0,
               'inbfrq':-1, 'imgfrq':-1}

    nbonds_dict['switch'] = True
    nbonds_dict['vfswitch'] = True
    nbonds_dict['ewald'] = True
    nbonds_dict['pmewald'] = True
    nbonds_dict['kappa'] = 0.32
    nbonds_dict['fftx'] = pmegrid
    nbonds_dict['ffty'] = pmegrid
    nbonds_dict['fftz'] = pmegrid
    nbonds_dict['order'] = 6
    nbonds=pycharmm.NonBondedScript(**nbonds_dict)
    nbonds.run()
    energy.show()

    charmm_script("blade on")
    

    ######################################
    # SET UP BLOCK FOR REFERENCE LAMBDAS #
    ######################################
    lig1_lambda_value = lambda_values[args.rank]
    lig2_lambda_value = 1 - lig1_lambda_value

    # always clear your blocks
    charmm_script(f"BLOCK {n_ligs+1}\nclear\nEND\n")

    # block command must be inputted as an entire string into charmm_script
    charmm_script(f"""
    BLOCK {n_ligs+1}
       ! assign benzene to Block 2 and phenol to Block 3
       Call 2 sele resname MOLB end
       Call 3 sele resname MOLP end

       qldm theta
       lang temp {temperature} 
       !RX! phmd ph 7
       soft on   ! this turns on soft-cores
       pmel ex   ! this turns on PME

       ldin 1 1.0 0.0 5.0 0.0 5.0
       ldin 2 {lig1_lambda_value:.2f} 0.0 5.0 0.0 5.0 
       ldin 3 {lig2_lambda_value:.2f} 0.0 5.0 0.0 5.0

       adex 2 3  ! 2nd and 3rd blocks shouldn't see each other 

       rmla bond thet impr  ! dihedral is scaled by lambda
       msld 0 1 1 ffix !fnex @fnex
       msma

       ldbi 0

    END
    """)

    # loop through every frame and compute current energy
    current_energies = np.zeros(n_frames)
    dcd_file = CharmmFile(file_name=f"{path}/dcd/{dcd_filename}.dcd", file_unit=101, formatted=False, read_only=True)
    charmm_script('traj query unit 101')
    charmm_script(f"traj firstu 101 nunit 1 nocheck")


    for i_frame in range(n_frames):
        charmm_script(f"traj read  !! FRAME {i_frame}/{n_frames} (current)")
        charmm_script("energy blade")
        current_energies[i_frame] = get_energy_value("ENER") 

    dcd_file.close()



    ######################################
    # SET UP BLOCK FOR PERTURBED LAMBDAS #
    ######################################
    lig1_lambda_value = lambda_values[args.rank+1]
    lig2_lambda_value = 1 - lig1_lambda_value

    # always clear your blocks
    charmm_script(f"BLOCK {n_ligs+1}\nclear\nEND\n")

    # block command must be inputted as an entire string into charmm_script
    charmm_script(f"""
    BLOCK {n_ligs+1}
       ! assign benzene to Block 2 and phenol to Block 3
       Call 2 sele resname MOLB end
       Call 3 sele resname MOLP end

       qldm theta
       lang temp {temperature} 
       !RX! phmd ph 7
       soft on   ! this turns on soft-cores
       pmel ex   ! this turns on PME

       ldin 1 1.0 0.0 5.0 0.0 5.0
       ldin 2 {lig1_lambda_value:.2f} 0.0 5.0 0.0 5.0 
       ldin 3 {lig2_lambda_value:.2f} 0.0 5.0 0.0 5.0

       adex 2 3  ! 2nd and 3rd blocks shouldn't see each other 

       rmla bond thet impr  ! dihedral is scaled by lambda
       msld 0 1 1 ffix !fnex @fnex
       msma

       ldbi 0

    END
    """)

    # loop through every frame and compute current energy
    perturbed_energies = np.zeros(n_frames)
    dcd_file = CharmmFile(file_name=f"{path}/dcd/{dcd_filename}.dcd", file_unit=101, formatted=False, read_only=True)
    charmm_script('traj query unit 101')
    charmm_script(f"traj firstu 101 nunit 1 nocheck")


    for i_frame in range(n_frames):
        charmm_script(f"traj read  !! FRAME {i_frame}/{n_frames} (perturbed)")
        charmm_script("energy blade")
        perturbed_energies[i_frame] = get_energy_value("ENER") 

    dcd_file.close()

    delta_energies = perturbed_energies - current_energies

    np.save(f"{path}/output/forward_delta_energies_w{args.rank}.npy", delta_energies)
    out_file.close()


if __name__ == "__main__":
    main()

