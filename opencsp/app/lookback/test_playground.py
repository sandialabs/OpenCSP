import re

strings = [
    "beginning_001_batch_0001.npz",
    "beginning_001_batch_0003.npz",
    "beginning_001_batch_0010.npz",
    "beginning_001_batch_0036.npz",
    "beginning_001_batch_0005.npz",
]

batch_pattern = r"batch_[0-9][0-9][0-9][0-9]"
batch_number = []
for batch in strings:
    temp = re.search(batch_pattern, batch)
    batch_number.append(temp.group(0))
    batch_test = re.search(batch_pattern, batch).group(0)

print(batch_number)
