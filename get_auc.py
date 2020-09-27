
from sklearn.metrics import accuracy_score, auc
import math
training_sizes = [160, 640, 2558, 10234, 40398]
x = [math.log2(t) for t in training_sizes]
x_diff = max(x)-min(x)

y = [61.6,	67.3,	72.1,	78.3,	81]

print(auc(x, y)/x_diff  )
