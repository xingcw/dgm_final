#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH -N 1                # Number of nodes requested.
#SBATCH -n 1                # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=8   # Number of cores per task.
#SBATCH --gres=gpu:l40s:1   # Number of GPUs.
#SBATCH -t 0-12:00:00                # Time requested (D-HH:MM).
#SBATCH --mem=32G                    # Memory requested.
#SBATCH --nodelist=al-l40s-0.grasp.maas    # Uncomment if you need a specific machine.
#SBATCH --partition=aloque-compute
#SBATCH --qos=al-high-2gpu

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
#SBATCH -D /mnt/aloque_scratch/chxing/git_repos/dgm_final

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.

#SBATCH -o ./tmp/slurm.%N.%j.out    # STDOUT
#SBATCH -e ./tmp/slurm.%N.%j.err    # STDERR

# Print some info for context.
pwd
hostname
date

ulimit -n 100000

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

srun PnPInversion/eval_script.sh