CPU nodes x 36. Every node has: 88 logical CPUs (86 usable) 500 GB RAM
GPU nodes:
   P40 nodes x 6 . Every node has: 
      24 CPUs (22 usable) 
      1 TB RAM NVIDIA Tesla P40 GPU x 1
   A100 nodes x 2 . Every node has: 
      128 CPUs (126 usable) 
      1 TB RAM NVIDIA Tesla A100 GPU x 8
maximum number of concurrently running jobs is 30,
maximum number of submitted jobs is 45.


QOS Name per Partition:
Batch partition: Use batch_default for QOS 
Debug partition: Use debug_default for QOS 
Serial partition: Use serial_default for QOS
GPU partition: Use gpu_p40_default for QOS
Account Name: pup-manila

CPU Hours: 20640

GPU Hours: 480

RAM: 250 GB

Storage: 100 GB

## this is what is shown when `squeue -u <coare username>` is ran in order to see the status of the submitted jobs
[michael.cueva@saliksik scratch1]$ squeue -u michael.cueva
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             25531     batch test_job michael. PD       0:00     36 (PartitionNodeLimit)
             25532     batch test_job michael. PD       0:00     36 (PartitionNodeLimit)
             25533     batch test_job michael. PD       0:00     36 (PartitionNodeLimit)
             25535     debug test_job michael. PD       0:00     10 (PartitionNodeLimit)
[michael.cueva@saliksik scratch1]$

because jobs are basically a set of instructions aka scripts that you want submit to the computing resource i.e. the HPC which is used by multiple people also trying to submit jobs that run potentially the scripts they want to run with haste using the computing resources they also requested and allocated for them

basically we want to create job scripts, submit the job scripts, depending on the number of people trying to submit jobs wait until our submitted job is processed, ran, and then finished, and then have the output in our hands i.e. a job.out file


this is our `test_job.sbatch` file

```
#!/bin/bash
#SBATCH --account=pup-manila    
#SBATCH --partition=debug
#SBATCH --qos=debug_default
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --job-name="test_job"
#SBATCH --output="%x.out"         ## <jobname>.<jobid>.out
##SBATCH --mail-type=ALL          ## optional
##SBATCH --mail-user=<email_add>  ## optional
##SBATCH --requeue                ## optional
##SBATCH --ntasks-per-node=1      ## optional
##SBATCH --mem=24G                ## optional: mem per node
##SBATCH --error="%x.%j.err"      ## optional; better to use --output only

## For more `sbatch` options, use `man sbatch` in the HPC, or go to https://slurm.schedmd.com/sbatch.html.

## Set stack size to unlimited.
ulimit -s unlimited

## Benchmarking.
start_time=$(date +%s.%N)

## Print job parameters.
echo "Submitted on $(date)"
echo "JOB PARAMETERS"
echo "SLURM_JOB_ID          : ${SLURM_JOB_ID}"
echo "SLURM_JOB_NAME        : ${SLURM_JOB_NAME}"
echo "SLURM_JOB_NUM_NODES   : ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST    : ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS          : ${SLURM_NTASKS}"
echo "SLURM_NTASKS_PER_NODE : ${SLURM_NTASKS_PER_NODE}"
echo "SLURM_MEM_PER_NODE    : ${SLURM_MEM_PER_NODE}"

## Create a unique temporary folder in the node. Using a local temporary folder usually results in faster read/write for temporary files.
custom_tmpdir="yes"

if [[ $custom_tmpdir == "yes" ]]; then
   JOB_TMPDIR=/tmp/${USER}/SLURM_JOB_ID/${SLURM_JOB_ID}
   mkdir -p ${JOB_TMPDIR}
   export TMPDIR=${JOB_TMPDIR}
   echo "TMPDIR                : $TMPDIR"
fi

## Reset modules.
module purge
module load anaconda/3-2023.07-2

## Main job. Run your codes and executables here; `srun` is optional.
conda activate thesis-writing-1
python --version
srun python "./thesis-writing-1/test.py"

## Flush the TMPDIR.
if [[ $custom_tmp == "yes" ]]; then
   rm -rf $TMPDIR
   echo "Cleared the TMPDIR (${TMPDIR})"
fi

## Benchmarking
end_time=$(date +%s.%N)
echo "Finished on $(date)"
run_time=$(python -c "print($end_time - $start_time)")
echo "Total runtime (sec): ${run_time}"
```

and when we submitted it via `sbatch test_job.sbatch` it will always just normally say `submitted job <certain job number>` because this indicates that other people using the computing resources have submitted a job script of their in order to run their individual scripts also, which therefore means we have to wait for our turn. And when it is our turn the job script is finally run and the instructions inside it indicating the scripts we want to run are then run, resulting in a `.out` file akin to the interface of a command line output where we run our scripts normally and get certain outputs this `.out` file also contains such outputs when the HPC finally runs our job scripts and therefore the individual scripts written inside of it 

```
Submitted on Fri Sep 13 18:37:50 PST 2024
JOB PARAMETERS
SLURM_JOB_ID          : 25534
SLURM_JOB_NAME        : test_job
SLURM_JOB_NUM_NODES   : 10
SLURM_JOB_NODELIST    : saliksik-cpu-[20,25-33]
SLURM_NTASKS          : 10
SLURM_NTASKS_PER_NODE : 
SLURM_MEM_PER_NODE    : 
TMPDIR                : /tmp/michael.cueva/SLURM_JOB_ID/25534
Module unload: anaconda/3-2023.07-2
/var/spool/slurm/job25534/slurm_script: line 5: conda: command not found
Module load: anaconda/3-2023.07-2
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Setting TMPDIR to /tmp
slurmstepd: error: Unable to create TMPDIR [/tmp/michael.cueva/SLURM_JOB_ID/25534]: No such file or directory
slurmstepd: error: Setting TMPDIR to /tmp
testing
testing
testing
testing
testing
testing
testing
testing
testing
testing
Finished on Fri Sep 13 18:37:52 PST 2024
Total runtime (sec): 2.068770170211792
```

`scancel <job_id>` to cancel a job
`squeue -u <coare username>` to see all queued and running jobs
`scontrol show job <job_id>` to show status, info, parameters of a queued or running job

```
[michael.cueva@saliksik thesis-writing-1]$ scontrol show job 26478
JobId=26478 JobName=tuning_ml_job
   UserId=michael.cueva(704002915) GroupId=michael.cueva(704002915) MCS_label=N/A
   Priority=1 Nice=0 Account=pup-manila QOS=batch_default
   JobState=PENDING Reason=QOSMaxJobsPerUserLimit Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=00:00:00 TimeLimit=7-00:00:00 TimeMin=N/A
   SubmitTime=2024-09-20T13:54:55 EligibleTime=2024-09-20T13:54:55
   AccrueTime=2024-09-20T13:54:55
   StartTime=Unknown EndTime=Unknown Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2024-09-21T12:38:09
   Partition=batch AllocNode:Sid=saliksik:41959
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=(null)
   NumNodes=20-20 NumCPUs=30 NumTasks=30 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=30,mem=30G,node=20,billing=30
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryCPU=1G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/hossain_gbt_tuning_job.sbatch
   WorkDir=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling
   StdErr=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/tuning_ml_job.out
   StdIn=/dev/null
   StdOut=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/tuning_ml_job.out
   Power=
```

```
scontrol show job 26090
JobId=26090 JobName=tuning_ml_job
   UserId=michael.cueva(704002915) GroupId=michael.cueva(704002915) MCS_label=N/A
   Priority=1 Nice=0 Account=pup-manila QOS=batch_default
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=1-16:30:42 TimeLimit=7-00:00:00 TimeMin=N/A
   SubmitTime=2024-09-19T20:09:42 EligibleTime=2024-09-19T20:09:42
   AccrueTime=2024-09-19T20:09:42
   StartTime=2024-09-19T20:09:52 EndTime=2024-09-26T20:09:52 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2024-09-19T20:09:52
   Partition=batch AllocNode:Sid=saliksik:17892
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=saliksik-cpu-[01-09,14-16,26-27,31-36]
   BatchHost=saliksik-cpu-01
   NumNodes=20 NumCPUs=20 NumTasks=20 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=20,mem=20G,node=20,billing=20
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryCPU=1G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/tuning_ml_job.sbatch
   WorkDir=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling
   StdErr=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/tuning_ml_job.out
   StdIn=/dev/null
   StdOut=/scratch1/michael.cueva/thesis-writing-1/server-side/modelling/tuning_ml_job.out
   Power=
```

So submitted job i.e. hossain_lr_tuning_job.sbatch works fine since it has outputted the reduced feature set file named `reduced_hossain_lr_feature_set.txt` as a by product 

```
[michael.cueva@saliksik Artifact Detection Data]$ stat reduced_hossain_lr_feature_set.txt
  File: ‘reduced_hossain_lr_feature_set.txt’
  Size: 766             Blocks: 8          IO Block: 4194304 regular file
Device: bd804632h/3179300402d   Inode: 144115514912852158  Links: 1
Access: (0664/-rw-rw-r--)  Uid: (704002915/michael.cueva)   Gid: (704002915/michael.cueva)
Access: 2024-09-23 08:17:29.000000000 +0800
Modify: 2024-09-21 19:09:28.000000000 +0800
Change: 2024-09-21 19:09:28.000000000 +0800
 Birth: -
[michael.cueva@saliksik Artifact Detection Data]$
```

limit of submitted jobs can only be 45
limit of concurrently running jobs can only be 30