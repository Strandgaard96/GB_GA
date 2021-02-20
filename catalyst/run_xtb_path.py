import os
import time

def slurm_xtb_path(reactant_xyz, product_xyz, inp_file, destination):
    org_dict = os.getcwd()
    os.chdir(destination)
    jobid = os.popen('submit_xtb_path_juls ' + reactant_xyz + ' ' + product_xyz + ' ' + str(inp_file)).read()
    jobid = int(jobid.split()[-1])
    os.chdir(org_dict)
    return jobid

def wait_for_jobs_to_finish(job_ids):
    """
    This script checks with slurm if a specific set of jobids is finished with a
    frequency of 1 minute.
    Stops when the jobs are done.
    """
    while True:
        job_info1 = os.popen("squeue -p coms").readlines()[1:]
        job_info2 = os.popen("squeue -u julius").readlines()[1:]
        current_jobs1 = {int(job.split()[0]) for job in job_info1}
        current_jobs2 = {int(job.split()[0]) for job in job_info2}
        current_jobs = current_jobs1|current_jobs2
        if current_jobs.isdisjoint(job_ids):
            break
        else:
            time.sleep(10)

def get_energy_from_path_ts(ts_path_file):
    with open(ts_path_file, 'r') as _file:
        for i, line in enumerate(_file):
            if i == 1:
                energy = float(line.split(' ')[2])
                break
    return energy