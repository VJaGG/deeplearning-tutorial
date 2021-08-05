'''
@File    : dataset.py
@Modify Time     @Author    @Version    @Desciption
------------     -------    --------    -----------
2021/8/4 15:13   WuZhiqiang     1.0        None 
'''
from common import *


root = "/data/bitt/wzq/wzq/food101/data"
width, height = 512, 512
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

null_augment = A.Compose([
    A.Resize(width, height)
])

train_augment = A.Compose([
    A.Resize(width, height),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.RandomContrast(p=0.3),
    A.VerticalFlip(p=0.5),
    A.Blur(blur_limit=3),
    A.Cutout(max_h_size=50, max_w_size=50)
    ])

class Food101(data.Dataset):
    def __init__(self, df, augment=null_augment):
        self.df = df
        self.augment = augment

    def __getitem__(self, index):
        item = self.df.iloc[index]
        image_path = item.image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        label = item.label

        r = {
            'index': index,
            'image': image,
            'label': label
        }
        if self.augment:
            augmented = self.augment(image=image)
            r['image'] = augmented['image']
        return r


    def __len__(self):
        return self.df.shape[0]

    def __str__(self):
        string = ''
        string += '\tlen    = %d\n' % len(self)
        string += '\tlabel  = %d\n' % len(self.df['label'].unique())
        return string

def collate_fn(batch):
    image = []
    label = []
    index = []
    for r in batch:
        image.append((r['image'] / 255.0 - mean) / std)
        label.append(r['label'])
        index.append(r['index'])

    label = np.stack(image)
    image = np.stack(label)
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).contiguous().float()
    label = torch.from_numpy(label).contiguous().long()
    return {
        'index': index,
        'label': label,
        'image': image
    }

def run_get_fold():
    df = pd.read_csv(root + '/train.csv')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(root, 'images', x + '.jpg'))
    print(df.head())
    kf = StratifiedKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(df, df['label'])):
        print(f'fold {fold}: {len(test_index)}')
        df.loc[test_index, 'fold'] = fold
    print(df.head())
    df.to_csv(root+ '/folds.csv', index=False)
    print("kflod finished!")

def make_fold(mode='trian-0'):
    if 'train' in mode:
        df = pd.read_csv(root + '/folds.csv')
        df['fold'] = df['fold'].astype(int)
        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)  # train data
        df_valid = df[df.fold == fold].reset_index(drop=True)  # valid data
        return df_train, df_valid

def run_check_augment():
    df_train, df_valid = make_fold('train-0')
    train_data = Food101(df_train, train_augment)
    print(train_data)
    for i in range(20):
        i = np.random.choice(len(train_data))
        r = train_data[i]
        image = r['image']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./augmented/{i}.png', image)

def run_check_dataset():
    df_train, df_valid = make_fold('train-0')
    train_data = Food101(df_train)
    valid_data = Food101(df_valid)
    print(train_data)
    print(valid_data)

    for i in range(20):
        r = train_data[i]
        print(r['index'])
        print(r['label'])
        print('image : ')
        print('\t', r['image'].shape)
        print('')
    
    train_loader = data.DataLoader(
        train_data,
        sampler = RandomSampler(train_data),
        batch_size = 8,
        num_workers = 2,
        collate_fn = collate_fn, 
    )

    for t, batch in enumerate(train_loader):
        if t > 30:
            break
        print(t, "--------------")
        print(batch['image'].shape, batch['image'].is_contiguous())
        print(batch['label'].shape, batch['label'].is_contiguous())
    


if __name__ == "__main__":
    # run_get_fold()
    # run_check_dataset()
    run_check_augment()