
from enum import Enum
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.pyplot import subplots, close, savefig
from os import error
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree,metrics
import random

DATASET_LENGTH = 67557
DEFAULT_STR = 'o,x,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,o,x,b,b,b,x,b,b,b,b,b,win'
CHAR_TO_NUM = {
	'b': 0,
	'o': 1,
	'x': -1,
}
#transform the dataset because the sklearn lib didnt support the current type
NUM_TO_CHAR = {
	0: ' ',
	1: 'o',
	-1: 'x',
}
RATIOS = [[40,60], [60,40], [80,20], [90,10]]
DATASET_BASE = [
	{
		"ratio": [40, 60],
		"filename_train": 'feature_label_train_4_6.dat',
		"filename_test": 'feature_label_test_4_6.dat',
		"filename_decision_tree": 'decision_tree_4_6',
		"filename_confusion_matrix": 'confusion_matrix_4_6'
	},
	{
		"ratio": [60,40],
		"filename_train": 'feature_label_train_6_4.dat',
		"filename_test": 'feature_label_test_6_4.dat',
		"filename_decision_tree": 'decision_tree_6_4',
		"filename_confusion_matrix": 'confusion_matrix_6_4'
	},
	{
		"ratio": [80, 20],
		"filename_train": 'feature_label_train_8_2.dat',
		"filename_test": 'feature_label_test_8_2.dat',
		"filename_decision_tree": 'decision_tree_8_2',
		"filename_confusion_matrix": 'confusion_matrix_8_2'
	},
	{
		"ratio": [90,10],
		"filename_train": 'feature_label_train_9_1.dat',
		"filename_test": 'feature_label_test_9_1.dat',
		"filename_decision_tree": 'decision_tree_9_1',
		"filename_confusion_matrix": 'confusion_matrix_9_1'
	},

]

LABEL_NODES = ['A1', ' A2', ' A3', ' A4', ' A5', ' A6', ' A7', ' A8', ' A9', ' A10', ' A11', ' A12', ' A13', ' A14', ' A15', ' A16', ' A17', ' A18', ' A19', ' A20', ' A21', ' A22', ' A23', ' A24', ' A25', ' A26', ' A27', ' A28', ' A29', ' A30', ' A31', ' A32', ' A33', ' A34', ' A35', ' A36', ' A37', ' A38', ' A39', ' A40', ' A41', ' A42']
#Assign the name of column so i can handle the data easily
class cellState(Enum):
    x='x'
    o='o'
    b='b'
class FourConnectResult(Enum):
	win = 'win'
	loss = 'loss'
	draw = 'draw'
	undefined = 'undefined'

def gettAllLineFromDataset(path=''):
    file=open('connect4.data','r')
    lines=file.readlines()
    file.close()
    return lines

class FourConnectState:

  def __init__(self, str = ''):
    self.parse(str.strip('\n'))

    if len(str.split(',')) != 43:
      # # Validate: eleminate wrong data
      # print('WRONG DATA: ' + str)
      self.parse(DEFAULT_STR)

  def parse(self, str):
    cells = str.split(',')
    parsedCells = list(map(lambda cell: CHAR_TO_NUM.get(cell), cells))
    self.state = parsedCells[ : -1]
    self.result = cells[len(cells) - 1]


  def show(self):
    for i in range(0, 6):
        row = []
        for j in range(0, 7):
            index = 6*(j+1) - i - 1
            row.append(NUM_TO_CHAR[self.state[index]])
        print(row)

    print(self.result)


print('Start spliting dataset')
#split the data into each ratio 
all_data = gettAllLineFromDataset('connect4.data')
random.shuffle(all_data)
for dataset in DATASET_BASE:
    filename_train = dataset["filename_train"]
    filename_test = dataset["filename_test"]
    ratio = dataset["ratio"]

    index = int(ratio[0] / 100 * len(all_data))

    file = open(filename_train, "w")
    file.write(''.join(all_data[ :index]))
    file.close()


    file = open(filename_test, "w")
    file.write(''.join(all_data[index: ]))
    file.close()
        

#process to accuracy, confusion matrix and decision tree with each ratio and max_depth
def process(
	dataset=DATASET_BASE[2],
	max_depth=None, 
	plot_decision_tree=False,
	plot_confusion_matrix=False):

	print('Running on ' +  str(dataset['ratio']) + ' dataset...',)
	print('With max_depth = ' + str(max_depth))
	training_set = gettAllLineFromDataset(dataset["filename_train"])
	testing_set = gettAllLineFromDataset(dataset["filename_test"])


	training_set_states = list(map(lambda str: FourConnectState(str=str), training_set))

	X_train = list(map(lambda state: state.state, training_set_states))
	Y_train = list(map(lambda state: state.result, training_set_states))

	decision_tree = tree.DecisionTreeClassifier(criterion="entropy",max_depth=max_depth,splitter='random').fit(X_train, Y_train)
	print('max_depth', decision_tree.tree_.max_depth)



	testing_set_states = list(map(lambda str: FourConnectState(str=str), testing_set))
	
	
	X_test = list(map(lambda state: state.state, testing_set_states))
	Y_test = list(map(lambda state: state.result, testing_set_states))

	Y_test_pred = decision_tree.predict(X_test)

	accuracy_score = metrics.accuracy_score(Y_test, Y_test_pred)

	print('Accuracy: ', accuracy_score)


	if plot_confusion_matrix:
		print('Start plotting confusion matrix phase...')
		metrics.ConfusionMatrixDisplay.from_predictions(Y_test, Y_test_pred, labels=decision_tree.classes_)
		plt.savefig(dataset["filename_confusion_matrix"] + '.png')
		print(metrics.classification_report(Y_test, Y_test_pred, labels=decision_tree.classes_,zero_division=0))

    
	if plot_decision_tree:
		print('Start plotting tree phase...')
		dot_data = tree.export_graphviz(
			filled=True,
			rounded=True,
			max_depth=max_depth,
			feature_names=LABEL_NODES,
			decision_tree=decision_tree
		)
		graph = graphviz.Source(dot_data) 
		graph.render(filename=dataset["filename_decision_tree"] + "_with_maxdepth_" + str(max_depth),format='png')

	return (accuracy_score)
#change the max_depth if you want, change plot_confusion_matrix to True so the program can create confusion matrix, same as decision tree
for dataset in DATASET_BASE:
	process(dataset=dataset,max_depth=None, plot_confusion_matrix=False, plot_decision_tree=False)
	print('\n')