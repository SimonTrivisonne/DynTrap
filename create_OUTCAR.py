
import re
import os


def create_str(t):
    """ create the '----- Ionic step        #  ------' string for a given t"""
    return f"--------------------------------------- Ionic step{' '*(9-len(str(t)))}{t}  -------------------------------------------\n"


def swap(l, pos1, pos2):
    l[pos1], l[pos2] = l[pos2], l[pos1]
    return l


if __name__ == "__main__":
    with open("OUTCAR_FINAL", "w") as outcar_final:
        l = sorted([int(re.search(r"\d+", path).group()) for path in os.listdir("OUTCARS")])
        EoI_time_dict = {}
        EoI_intern_time_list = []
        with open("polaron_dynamic_log.txt", "r") as log:
            for i, line in zip(l, log.readlines()):
                if re.search(r'no polaron jump detected', line):
                    EoI_time_dict[i] = [-1, -1]
                    # get max timestepsMD from the job.sh
                    with open('job.sh') as job:
                        j = job.read()
                        timestepsmd = int(re.findall(r'(?<=timestepsMD=)\d+', j)[0])
                    EoI_intern_time_list.append(timestepsmd-1)    # this should match the NSW in INCAR - 1
                else:
                    EoI_time_dict[i] = list(map(int, re.findall(r'(?<=Ti)\d+', line)))
                    EoI_intern_time_list.append(int(re.search(r"(?<=Time=)\d+", line).group()))
        EoI_time_list = list(EoI_time_dict.keys())

        for t_intern, t in zip(EoI_intern_time_list, EoI_time_list):
            with open(f"OUTCARS/OUTCAR_{t}", "r") as outcar:
                o = outcar.readlines()
                start_at = o.index(create_str(1))
                if t == EoI_time_list[0]:
                    # insert header from first OUTCAR
                    header = o[:start_at]
                    outcar_final.write("".join(header))
                if t != EoI_time_list[-1]:
                    end_at = o.index(create_str(t_intern + 1))
                else:
                    with open('job.sh') as job:
                        j = job.read()
                        f = re.findall(r'timesteps *= *\d+', j)[0]
                        timesteps = int(re.findall(r'\d+', f)[0])
                    end_at = o.index(create_str(t_intern + 1 - (t - timesteps)))

                """ update OUTCAR_FINAL so that atoms ids matches the initial POSCAR """
                if t != EoI_time_list[0]:
                    for j, line in enumerate(o[start_at:end_at]):
                        if line == create_str(t_intern + 1):
                            break
                        if line == '  POSITION                                       TOTAL-FORCE (eV/Angst) (ML)\n':
                            i = EoI_time_list.index(t)
                            EoI_time_list.reverse()
                            for e in EoI_time_list[-i:]:
                                pj = EoI_time_dict[e]
                                o = swap(o, start_at + j + pj[0] + 1, start_at + j + pj[1] + 1)
                            EoI_time_list.reverse()
                outcar_final.write("".join(o[start_at:end_at]))

    with open("OUTCAR_FINAL", "r+") as outcar_final:
        # re-numerate the Ionic steps
        o = outcar_final.readlines()
        count = 1
        for i, line in enumerate(o):
            if re.search(r'-+ Ionic step +\d+ +-+\n', line):
                o[i] = re.sub(line, create_str(count), line)
                count += 1
        outcar_final.seek(0)
        outcar_final.truncate()
        outcar_final.write("".join(o))
