from ppo import main
import sys

n = 64
if len(sys.argv) > 1:
    n = int(sys.argv[1])

print(f"Executing ppo with random floorplan of size {n}")
main(n)
