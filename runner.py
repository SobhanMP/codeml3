from main import Clf, parser
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd
import wandb
import tempfile
args = parser.parse_args()
clf = Clf(project="cross-run", **vars(args))
df = pd.read_csv('data/participant_urbansound8k.csv')
s = df.Label.isnull()
train_df = pd.DataFrame(df[~s].reset_index())
train_df.Label = (train_df.Label == True)
test_df = df[s].reset_index()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
total_f1 = 0
total_f1_n = 0
for train_index, test_index in cv.split(train_df, train_df.Label):
    local_train_df = train_df.iloc[train_index]
    local_test_df = train_df.iloc[test_index]
    clf.fit(local_train_df)
    pred = clf.pred(local_test_df)
    f1 = f1_score(local_test_df.Label, pred)
    print(f'f1: {f1}, {total_f1}, {total_f1_n}')
    s = len(local_test_df) + total_f1_n
    total_f1 = total_f1_n / s * total_f1 + len(local_test_df) / s * f1
    total_f1_n = s
    wandb.finish()
clf.project="hyper-run"
clf.fit(train_df, train_df.Label)
res = test_df.copy()
res.Label = clf.pred(test_df)
with tempfile.NamedTemporaryFile() as temp:
    res.to_csv(temp.name, index=False)
    wandb.save(temp.name)
    wandb.log({
        'test_score': total_f1
    })
    wandb.finish()