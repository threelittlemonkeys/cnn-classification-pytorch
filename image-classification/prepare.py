from utils import *

def load_data():
    tti = {} # tag_to_idx
    fo = open(sys.argv[1])
    for line in fo:
        x, y = line.split("\t")
        y = y.strip()
        if y not in tti:
            tti[y] = len(tti)
    fo.close()
    return tti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s img_to_tag" % sys.argv[0])
    tti = load_data()
    save_tkn_to_idx("tag_to_idx", tti)
