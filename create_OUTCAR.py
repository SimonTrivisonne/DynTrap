# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Merge the OUTCARs of all MD segments into one continuous OUTCAR (``OUTCAR_FINAL``).

DynTrap runs the MD in segments. Every segment ends at a polaron jump, and the next segment is started
from a POSCAR in which the two jumping Ti were swapped. This script stitches the segments back together:

    1. Read which OUTCAR belongs to which jump from ``polaron_dynamic_log.txt``.
    2. Cut every OUTCAR at the step of its jump (the last one at the end of the simulation).
    3. Undo the Ti swaps of all earlier segments in the atom order of every ionic step, so that the atom
       ids match the initial POSCAR again.
    4. Renumber the ionic steps of the merged file from 1.

Input (in the current directory)
    OUTCARS/OUTCAR_<n>          one OUTCAR per segment; <n> is the first number in the file name
    polaron_dynamic_log.txt     one line per segment, in the order of the sorted <n> (written by DynTrap)
    job.sh                      contains ``timestepsMD=<int>`` and ``timesteps = <int>``
Output
    OUTCAR_FINAL

Assumptions: the numbers <n> are unique, and the log has one line for every OUTCAR (if not, the shorter
of the two decides how many segments are merged).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ======================================================================================================
# Configuration
# ======================================================================================================

OUTCAR_DIR = "OUTCARS"
OUTPUT_FILE = "OUTCAR_FINAL"
LOG_FILE = "polaron_dynamic_log.txt"
JOB_SCRIPT_FILE = "job.sh"

# Line that starts the list of atom positions of an ionic step (the atom rows follow after one more line)
POSITION_HEADER_LINE = '  POSITION                                       TOTAL-FORCE (eV/Angst) (ML)\n'

# Matches the '----- Ionic step        #  ------' line of an ionic step
IONIC_STEP_PATTERN = re.compile(r'-+ Ionic step +\d+ +-+\n')

NO_JUMP = [-1, -1]                     # "swapped Ti" of a segment that ended without a polaron jump


@dataclass
class MdSegment:
    """One MD segment, i.e. one OUTCAR and the polaron jump that ended it."""
    number: int                        # <n> of OUTCARS/OUTCAR_<n>
    swapped_ti: list                   # [Ti_from, Ti_to] (1-based) of the jump, or NO_JUMP
    last_step: int                     # ionic step of the jump; the segment is cut after this step


# ======================================================================================================
# Helpers
# ======================================================================================================

def create_ionic_step_header(step: int) -> str:
    """Create the '----- Ionic step        #  ------' line for a given step number."""
    return f"--------------------------------------- Ionic step{step:>9}  -------------------------------------------\n"


def swap(items: list, pos1: int, pos2: int) -> None:
    """Swap two entries of a list in place."""
    items[pos1], items[pos2] = items[pos2], items[pos1]


def read_timesteps_md() -> int:
    """Number of MD steps of one segment: ``timestepsMD=<int>`` in job.sh."""
    with open(JOB_SCRIPT_FILE) as job:
        job_script = job.read()
    return int(re.findall(r'(?<=timestepsMD=)\d+', job_script)[0])


def read_timesteps() -> int:
    """``timesteps = <int>`` in job.sh."""
    with open(JOB_SCRIPT_FILE) as job:
        job_script = job.read()
    entry = re.findall(r'timesteps *= *\d+', job_script)[0]
    return int(re.findall(r'\d+', entry)[0])


# ======================================================================================================
# Step 1: read the segments from the log
# ======================================================================================================

def read_segments() -> list[MdSegment]:
    """Pair every OUTCAR (sorted by number) with the corresponding line of the DynTrap log."""
    outcar_numbers = sorted(int(re.search(r"\d+", name).group()) for name in os.listdir(OUTCAR_DIR))
    segments = []
    with open(LOG_FILE, "r") as log:
        for number, line in zip(outcar_numbers, log.readlines()):
            if re.search(r'no polaron jump detected', line):
                swapped_ti = list(NO_JUMP)
                # the segment ran to its last MD step
                last_step = read_timesteps_md() - 1    # this should match the NSW in INCAR - 1
            else:
                # log line: "polaron jump from Ti<from> to Ti<to> at Time=<step>"
                swapped_ti = list(map(int, re.findall(r'(?<=Ti)\d+', line)))
                last_step = int(re.search(r"(?<=Time=)\d+", line).group())
            segments.append(MdSegment(number=number, swapped_ti=swapped_ti, last_step=last_step))
    return segments


# ======================================================================================================
# Step 2-3: cut the OUTCARs and restore the atom order
# ======================================================================================================

def find_end_of_segment(lines: list[str], segment: MdSegment, is_last: bool) -> int:
    """Line index where the part of this OUTCAR that is kept ends (exclusive)."""
    if not is_last:
        # keep everything up to (not including) the step after the jump
        return lines.index(create_ionic_step_header(segment.last_step + 1))
    # the last segment is cut at an end step that is shifted by (segment number - timesteps from job.sh)
    shift = segment.number - read_timesteps()
    return lines.index(create_ionic_step_header(segment.last_step + 1 - shift))


def restore_atom_order(lines: list[str], start_at: int, end_at: int, segment: MdSegment,
                       earlier_swaps: list[list]) -> None:
    """
    Undo the Ti swaps of all earlier segments in every ionic step of ``lines[start_at:end_at]`` (in place).

    The POSITION block of each ionic step is the header line followed by one more line and then one row
    per atom, so atom k (1-based) sits k + 1 lines below the header. The most recent swap is undone first.
    """
    stop_header = create_ionic_step_header(segment.last_step + 1)
    for offset, line in enumerate(lines[start_at:end_at]):
        if line == stop_header:
            break
        if line == POSITION_HEADER_LINE:
            header_index = start_at + offset
            for ti_from, ti_to in reversed(earlier_swaps):
                swap(lines, header_index + ti_from + 1, header_index + ti_to + 1)


def write_merged_outcar(segments: list[MdSegment]) -> None:
    """Write OUTCAR_FINAL: header of the first OUTCAR + the kept ionic steps of every segment."""
    with open(OUTPUT_FILE, "w") as outcar_final:
        for index, segment in enumerate(segments):
            is_first = index == 0
            is_last = index == len(segments) - 1

            with open(f"{OUTCAR_DIR}/OUTCAR_{segment.number}", "r") as outcar:
                lines = outcar.readlines()
            start_at = lines.index(create_ionic_step_header(1))

            if is_first:
                # the header (everything before the first ionic step) is taken from the first OUTCAR only
                outcar_final.write("".join(lines[:start_at]))

            end_at = find_end_of_segment(lines, segment, is_last)

            # update the segment so that the atom ids match the initial POSCAR
            if not is_first:
                earlier_swaps = [earlier.swapped_ti for earlier in segments[:index]]
                restore_atom_order(lines, start_at, end_at, segment, earlier_swaps)

            outcar_final.write("".join(lines[start_at:end_at]))


# ======================================================================================================
# Step 4: renumber the ionic steps
# ======================================================================================================

def renumber_ionic_steps(path: str) -> None:
    """Renumber the ionic steps of a merged OUTCAR consecutively from 1 (in place)."""
    with open(path, "r+") as outcar_final:
        lines = outcar_final.readlines()
        count = 1
        for index, line in enumerate(lines):
            if IONIC_STEP_PATTERN.search(line):
                lines[index] = create_ionic_step_header(count)
                count += 1
        outcar_final.seek(0)
        outcar_final.truncate()
        outcar_final.write("".join(lines))


def main():
    segments = read_segments()
    write_merged_outcar(segments)
    renumber_ionic_steps(OUTPUT_FILE)


if __name__ == "__main__":
    main()