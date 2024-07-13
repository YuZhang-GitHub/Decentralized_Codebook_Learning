import os
import subprocess
import multiprocessing

path = os.getcwd()
path = os.path.join(path, 'main.py')


def worker_0(GPU_ID, user_group):
    subprocess.run(('python', path, str(GPU_ID), str(user_group)))


if __name__ == '__main__':

    beam_id_set = list(range(16))

    process_list = []

    for ii in range(len(beam_id_set)):

        gpu_id = ii % 1

        process_list.append(multiprocessing.Process(target=worker_0, args=(gpu_id, beam_id_set[ii])))

        process_list[ii].start()

    for ii in range(len(beam_id_set)):

        process_list[ii].join()
