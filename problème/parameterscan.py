import numpy as np
import subprocess
import os

# Parameters
repertoire = ''
executable = './engine'
input_filename = 'configuration.in.example' # Strictly no longer needed, but we keep it for now to avoid having to change the code in engine.cpp


input_parameters = {
    'tf': 172800,
    'G': 0.00000000006674,
    'mA': 8500,
    'r0': 314159000,
    'd': 5.02,
    'v0': 1200,
    'h': 10000,
    'RT': 6378100,
    'mT': 5972000000000000000000000,
    'nsteps': 172800/2,
    'sampling': 1,
    'idx': 1
}

# -------------------------------------------------

# Updated from last time, the code below can now be used to scan any parameter, just make sure to update the paramstr and the variable_array accordingly

paramstr = 'nsteps' # The parameter to scan, must be one of the keys in input_parameters

variable_array = [50, 100, 200, 400, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Example values for the parameter scan
outstr = f"Trajectory_{input_parameters['nsteps']:.4g}"

# -------------------------------------------------
# Create output directory (2 significant digits)
# -------------------------------------------------
outdir = f"Scan_{paramstr}_{outstr}"
os.makedirs(outdir, exist_ok=True)
print("Saving results in:", outdir)


for i in range(len(variable_array)):

    # Copy parameters and overwrite scanned one
    params = input_parameters.copy()
    params[paramstr] = variable_array[i]

    output_file = f"{outstr}_{paramstr}_{variable_array[i]}.txt"
    output_path = os.path.join(outdir, output_file)

    # Build parameter string
    param_string = " ".join(f"{k}={v:.15g}" for k, v in params.items())

    cmd = (
        f"{repertoire}{executable} {input_filename} "
        f"{param_string} output={output_path}"
    )

    print(cmd)
    subprocess.run(cmd, shell=True)
    print("Done.")

