# POVa
# xpavlu10
"""Test dataset and dataloader
"""

from dataloader import make_dataloader
import pdb

if __name__ == "__main__":
    dataloader = make_dataloader(
        "./pero/train_copy.easy", "./lines", batch_size=100, shuffle=False, verbose=True)
    for images, labels in dataloader:
        pdb.set_trace()
