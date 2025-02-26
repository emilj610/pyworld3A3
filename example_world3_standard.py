# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

params = {"lines.linewidth": "3"}
plt.rcParams.update(params)

world3 = World3(year_max=2100)
world3.set_world3_control()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert],
    ["AL", "PAL", "UIL", "LFERT"],
    [[0, 25e8], [0, 23e8], [0, 120e6], [0, 800]],
    figsize=(7, 5),
    title="World3 control run - Agriculture",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.ic, world3.sc],
    ["IC", "SC"],
    [[0, 12e12], [0, 50e11]],
    figsize=(7, 5),
    title="World3 control run - Capital",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.ppol],
    ["PPOL"],
    [[0, 20e8]],
    figsize=(7, 5),
    title="World3 control run - Pollution",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.p1, world3.p2, world3.p3, world3.p4],
    ["P1", "P2", "P3", "P4"],
    [[0, 3e9], [0, 5e9], [0, 2e9], [0, 2e9]],
    figsize=(7, 5),
    title="World3 control run - Population",
)
plt.grid()

plot_world_variables(
    world3.time,
    [world3.nr],
    ["NR"],
    [[0, 1e12]],
    figsize=(7, 5),
    title="World3 control run - Resources",
)
plt.grid()
plt.show()
