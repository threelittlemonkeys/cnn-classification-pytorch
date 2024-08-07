from model import *
from utils import *
from dataloader import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    model = cnn(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt

def run_model(model, data, itt):
    with torch.no_grad():
        model.eval()
        for batch in data.split():
            xc, xw, lens = batch.xc, batch.xw, batch.lens
            xc, xw = data.tensor(bc = xc, bw = xw, lens = lens)
            y1 = model(xc, xw)
            batch.y1 = [itt[y.argmax()] for y in y1]
            batch.prob = [y.max().exp().item() for y in y1]
            for x0, y0, y1, prob in zip(batch.x0, batch.y0, batch.y1, batch.prob):
                yield *x0, *y0, y1, prob

def predict(model, cti, wti, itt, filename):
    data = dataloader()
    fo = open(filename)
    for line in fo:
        data.append_row()
        line = line.strip()
        x0, y0 = re.findall("(.+?)(?:\t(.+))?$", line)[0]
        x1 = list(map(normalize, tokenize(x0)))
        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x1]
        xw = [wti[w] if w in wti else UNK_IDX for w in x1]
        data.append_item(x0 = x0, x1 = x1, xc = xc, xw = xw, y0 = y0)
    fo.close()
    return run_model(model, data, itt)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    result = predict(*load_model(), sys.argv[5])
    for x, y0, y1, prob in result:
        print((x, y0, y1, prob))
