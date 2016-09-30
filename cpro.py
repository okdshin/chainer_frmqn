import sys
import pstats
p = pstats.Stats(sys.argv[1])
p.sort_stats('cumulative').print_stats(20)
