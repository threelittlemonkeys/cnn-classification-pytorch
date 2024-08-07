from model import *
from utils import *
from PIL import Image

def load_data(args):

    data = []
    batch = []
    tti = load_tkn_to_idx(args[2]) # tkn_to_tkn
    itt = load_idx_to_tkn(args[2]) # idx_to_tkn

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ])

    print(f"loading data")

    with open(args[1], "r") as fo:
        for line in fo:
            x, y = line.strip().split(" ")
            x = transform(Image.open(x))
            y = tti[y]
            data.append((x, y))

    for idx in range(0, len(data), BATCH_SIZE):
        x0, y0 = zip(*data[idx:idx + BATCH_SIZE])
        x0 = t_stack(x0)
        y0 = t_LongTensor(y0)
        batch.append((x0, y0))

    print(f"data size: {len(data)}")
    print(f"batch size: {BATCH_SIZE}")

    return batch, itt

def train(args):

    num_epochs = int(args[-1])
    batch, itt = load_data(args)
    model = cnn(len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)

    epoch = load_checkpoint(args[0], model) if isfile(args[0]) else 0
    filename = re.sub("\\.epoch[0-9]+$", "", args[0])

    print("training model")

    for ei in range(epoch + 1, epoch + num_epochs + 1):

        loss_sum = 0
        timer = time()

        for x0, y0 in batch:
            model.zero_grad()
            y1 = model(x0)
            loss = F.nll_loss(y1, y0) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()

        timer = time() - timer
        loss_sum /= len(batch)

        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)

        if len(args) == 7 and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            evaluate(predict(model, itt, args[5]), True)
            model.train()
            print()

if __name__ == "__main__":

    if len(sys.argv) not in [5, 6]:
        sys.exit("Usage: %s model img_to_tag tag_to_idx (validation_data) num_epoch" % sys.argv[0])

    train(sys.argv[1:])
