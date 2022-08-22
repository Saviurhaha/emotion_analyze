import glob
import pandas as pd
from tqdm import tqdm

system = 'Win'

train_pos_path = 'datasets/aclImdb/train/pos/*'
train_neg_path = 'datasets/aclImdb/train/neg/*'
train_pos = glob.glob(train_pos_path)
train_neg = glob.glob(train_neg_path)
test_pos_path = 'datasets/aclImdb/test/pos/*'
test_neg_path = 'datasets/aclImdb/test/neg/*'
test_pos = glob.glob(test_pos_path)
test_neg = glob.glob(test_neg_path)
train_df = []
test_df = []


def read_data(path, message):
    res = []
    for p in tqdm(path, desc=message, position=0):
        with open(p, encoding="utf8") as f:
            text = f.read()

            #         For win users
            if system == 'Win':
                beg = p.find('\\')

            #         For mac users
            # if system == 'Mac':
            #  beg = re.search(r"\d", p).start() - 1

            idx, rating = p[beg + 1:-4].split('_')
            res.append([text, rating])

    return res


train_df += read_data(path=train_pos, message='Getting positive train data')
train_df += read_data(path=train_neg, message='Getting negative train data')

test_df += read_data(path=test_pos, message='Getting positive test data')
test_df += read_data(path=test_neg, message='Getting negative test data')

train_df = pd.DataFrame(train_df, columns=['text', 'rating'])
test_df = pd.DataFrame(test_df, columns=['text', 'rating'])

print('Records: ', train_df.size)
train_df.head()

for i in range(1, 11):
    print(f'Number of reviews with rating {i}: {train_df[train_df.rating == str(i)].shape[0]}')
