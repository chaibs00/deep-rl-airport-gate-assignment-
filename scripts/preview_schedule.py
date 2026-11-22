"""
Preview a generated schedule.
    python -m scripts.preview_schedule
"""


import numpy as np
from collections import Counter
from gatemind.environement.airport import Airport
from gatemind.data.schedule_generator import ScheduleGenerator

airport = Airport()
gen = ScheduleGenerator(airport.sizes)
flights = gen.generate(np.random.default_rng(0))

def hhmm(m):
    return f"{int(m // 60):02d}:{int(m % 60):02d}"

for f in flights[:5]:
    print(f"#{f.flight_id:2d}  {f.category:8s}  arr {hhmm(f.arrival)}  dep {hhmm(f.departure)}  ({f.turnaround:.0f} min)")

print("Fleet:", dict(Counter(f.category for f in flights)))

# Busiest moment: how many aircraft are on the ground at the same time?
times = np.arange(360, 1440, 5)
on_ground = [sum(f.arrival <= t < f.departure for f in flights) for t in times]
peak = int(np.argmax(on_ground))
print(f"Max aircraft on the ground: {max(on_ground)} at {hhmm(times[peak])} (23 contact gates)")