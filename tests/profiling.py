import sys
import objgraph
from context import main

if __name__ == "__main__":
  main.main()
  with open('tmp/profile_output.txt', 'w') as f:
    sys.stdout = f
    objgraph.show_most_common_types(limit=0)
    print("".center(50, "#"))
    objgraph.show_growth(limit=0)
    sys.stdout = sys.__stdout__  # Reset standard output to default
  objgraph.show_refs([main.main], max_depth=10, filename='tmp/refs.png')
