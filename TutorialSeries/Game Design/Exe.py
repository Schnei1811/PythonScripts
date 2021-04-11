#https://pythonprogramming.net/converting-pygame-executable-cx_freeze/?completed=/pygame-button-function-events/

import cx_Freeze

executables = [cx_Freeze.Executable("Intro.py")]

cx_Freeze.setup(
    name="A bit Racey",
    options={"build_exe": {"packages":["pygame"],
                           "include_files":["racecar.png"]}},
    executables = executables

    )