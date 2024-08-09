from model import *
from utils import *
from PIL import Image

def load_model():

    itt = load_idx_to_tkn(sys.argv[2]) # idx_to_tag

    model = cnn(len(itt))
    print(model)

    load_checkpoint(sys.argv[1], model)

    return model, itt


def run_model(model, data, itt):

    with torch.no_grad():
        model.eval()

        for idx in range(0, len(data), BATCH_SIZE):
            x0, x1, y0 = zip(*data[idx:idx + BATCH_SIZE])
            x1 = t_stack(x1)
            y1 = model(x1)
            y1, prob = zip(*[max(enumerate(y), key = lambda x: x[1]) for y in y1])
            y1 = [itt[y] for y in y1]
            prob = [p.max().exp().item() for p in prob]
            for x0, y0, y1, prob in zip(x0, y0, y1, prob):
                yield x0, y0, y1, prob

def predict(model, itt, filename):

    data = []

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    with open(filename) as fo:
        for line in fo:
            x0, *y0 = line.strip().split("\t")
            y0 = y0[0] if y0 else ""
            x1 = transform(Image.open(x0))
            data.append((x0, x1, y0))

    return run_model(model, data, itt)

if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.exit("Usage: %s model tag_to_idx test_data" % sys.argv[0])

    result = predict(*load_model(), sys.argv[3])

    for x, y0, y1, prob in result:
        print((x, y0, y1, prob))
